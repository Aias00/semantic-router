//go:build milvus
// +build milvus

package cache

import (
	"context"
	"crypto/md5"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"gopkg.in/yaml.v3"
)

// MilvusConfig defines the complete configuration structure for Milvus cache backend
type MilvusConfig struct {
	Connection struct {
		Host     string `yaml:"host"`
		Port     int    `yaml:"port"`
		Database string `yaml:"database"`
		Timeout  int    `yaml:"timeout"`
		Auth     struct {
			Enabled  bool   `yaml:"enabled"`
			Username string `yaml:"username"`
			Password string `yaml:"password"`
		} `yaml:"auth"`
		TLS struct {
			Enabled  bool   `yaml:"enabled"`
			CertFile string `yaml:"cert_file"`
			KeyFile  string `yaml:"key_file"`
			CAFile   string `yaml:"ca_file"`
		} `yaml:"tls"`
	} `yaml:"connection"`
	Collection struct {
		Name        string `yaml:"name"`
		Description string `yaml:"description"`
		VectorField struct {
			Name       string `yaml:"name"`
			Dimension  int    `yaml:"dimension"`
			MetricType string `yaml:"metric_type"`
		} `yaml:"vector_field"`
		Index struct {
			Type   string `yaml:"type"`
			Params struct {
				M              int `yaml:"M"`
				EfConstruction int `yaml:"efConstruction"`
			} `yaml:"params"`
		} `yaml:"index"`
	} `yaml:"collection"`
	Search struct {
		Params struct {
			Ef int `yaml:"ef"`
		} `yaml:"params"`
		TopK             int    `yaml:"topk"`
		ConsistencyLevel string `yaml:"consistency_level"`
	} `yaml:"search"`
	Performance struct {
		ConnectionPool struct {
			MaxConnections     int `yaml:"max_connections"`
			MaxIdleConnections int `yaml:"max_idle_connections"`
			AcquireTimeout     int `yaml:"acquire_timeout"`
		} `yaml:"connection_pool"`
		Batch struct {
			InsertBatchSize int `yaml:"insert_batch_size"`
			Timeout         int `yaml:"timeout"`
		} `yaml:"batch"`
	} `yaml:"performance"`
	DataManagement struct {
		TTL struct {
			Enabled         bool   `yaml:"enabled"`
			TimestampField  string `yaml:"timestamp_field"`
			CleanupInterval int    `yaml:"cleanup_interval"`
		} `yaml:"ttl"`
		Compaction struct {
			Enabled  bool `yaml:"enabled"`
			Interval int  `yaml:"interval"`
		} `yaml:"compaction"`
	} `yaml:"data_management"`
	Logging struct {
		Level          string `yaml:"level"`
		EnableQueryLog bool   `yaml:"enable_query_log"`
		EnableMetrics  bool   `yaml:"enable_metrics"`
	} `yaml:"logging"`
	Development struct {
		DropCollectionOnStartup bool `yaml:"drop_collection_on_startup"`
		AutoCreateCollection    bool `yaml:"auto_create_collection"`
		VerboseErrors           bool `yaml:"verbose_errors"`
	} `yaml:"development"`
}

// MilvusCache provides a scalable semantic cache implementation using Milvus vector database
type MilvusCache struct {
	client              client.Client
	config              *MilvusConfig
	collectionName      string
	similarityThreshold float32
	ttlSeconds          int
	enabled             bool
	hitCount            int64
	missCount           int64
	lastCleanupTime     *time.Time
	mu                  sync.RWMutex
	cleanupStopChan     chan struct{}
	cleanupWG           sync.WaitGroup
	batchBuffer         []*CacheEntry
	batchMu             sync.Mutex
	batchTimer          *time.Timer
	writeMetrics        struct {
		totalWrites     int64
		batchWrites     int64
		singleWrites    int64
		writeErrors     int64
		avgWriteLatency float64
		lastWriteTime   time.Time
	}
}

// MilvusCacheOptions contains configuration parameters for Milvus cache initialization
type MilvusCacheOptions struct {
	SimilarityThreshold float32
	TTLSeconds          int
	Enabled             bool
	ConfigPath          string
}

// NewMilvusCache initializes a new Milvus-backed semantic cache instance
func NewMilvusCache(options MilvusCacheOptions) (*MilvusCache, error) {
	if !options.Enabled {
		observability.Debugf("MilvusCache: disabled, returning stub")
		return &MilvusCache{
			enabled: false,
		}, nil
	}

	// Load Milvus configuration from file
	observability.Debugf("MilvusCache: loading config from %s", options.ConfigPath)
	config, err := LoadMilvusConfig(options.ConfigPath)
	if err != nil {
		observability.Debugf("MilvusCache: failed to load config: %v", err)
		return nil, fmt.Errorf("failed to load Milvus config: %w", err)
	}
	observability.Debugf("MilvusCache: config loaded - host=%s:%d, collection=%s, dimension=auto-detect",
		config.Connection.Host, config.Connection.Port, config.Collection.Name)

	// Establish connection to Milvus server
	connectionString := fmt.Sprintf("%s:%d", config.Connection.Host, config.Connection.Port)
	observability.Debugf("MilvusCache: connecting to Milvus at %s", connectionString)
	milvusClient, err := client.NewGrpcClient(context.Background(), connectionString)
	if err != nil {
		observability.Debugf("MilvusCache: failed to connect: %v", err)
		return nil, fmt.Errorf("failed to create Milvus client: %w", err)
	}
	observability.Debugf("MilvusCache: successfully connected to Milvus")

	cache := &MilvusCache{
		client:              milvusClient,
		config:              config,
		collectionName:      config.Collection.Name,
		similarityThreshold: options.SimilarityThreshold,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
		cleanupStopChan:     make(chan struct{}),
		batchBuffer:         make([]*CacheEntry, 0, config.Performance.Batch.InsertBatchSize),
	}

	// Initialize batch write timer if batch size > 1
	if config.Performance.Batch.InsertBatchSize > 1 {
		cache.startBatchWriter()
	}

	// Set up the collection for caching
	observability.Debugf("MilvusCache: initializing collection '%s'", config.Collection.Name)
	if err := cache.initializeCollection(); err != nil {
		observability.Debugf("MilvusCache: failed to initialize collection: %v", err)
		milvusClient.Close()
		return nil, fmt.Errorf("failed to initialize collection: %w", err)
	}
	observability.Debugf("MilvusCache: initialization complete")

	// Start TTL cleanup goroutine if enabled
	if config.DataManagement.TTL.Enabled && options.TTLSeconds > 0 {
		observability.Debugf("MilvusCache: starting TTL cleanup goroutine (interval: %ds)", config.DataManagement.TTL.CleanupInterval)
		cache.startTTLCleanup()
	}

	return cache, nil
}

// LoadMilvusConfig reads and parses the Milvus configuration from file
func LoadMilvusConfig(configPath string) (*MilvusConfig, error) {
	if configPath == "" {
		return nil, fmt.Errorf("Milvus config path is required")
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config MilvusConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Validate configuration
	if err := validateMilvusConfig(&config); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	return &config, nil
}

// validateMilvusConfig validates the Milvus configuration parameters
func validateMilvusConfig(config *MilvusConfig) error {
	// Validate connection settings
	if config.Connection.Host == "" {
		return fmt.Errorf("connection host is required")
	}
	if config.Connection.Port <= 0 || config.Connection.Port > 65535 {
		return fmt.Errorf("connection port must be between 1 and 65535")
	}
	if config.Connection.Timeout <= 0 {
		config.Connection.Timeout = 30 // Default timeout
	}

	// Validate authentication
	if config.Connection.Auth.Enabled {
		if config.Connection.Auth.Username == "" {
			return fmt.Errorf("username is required when authentication is enabled")
		}
		if config.Connection.Auth.Password == "" {
			return fmt.Errorf("password is required when authentication is enabled")
		}
	}

	// Validate TLS configuration
	if config.Connection.TLS.Enabled {
		if config.Connection.TLS.CertFile == "" || config.Connection.TLS.KeyFile == "" {
			return fmt.Errorf("both cert_file and key_file are required when TLS is enabled")
		}
	}

	// Validate collection settings
	if config.Collection.Name == "" {
		config.Collection.Name = "semantic_cache" // Default collection name
	}
	if config.Collection.VectorField.Name == "" {
		config.Collection.VectorField.Name = "embedding" // Default vector field name
	}
	if config.Collection.VectorField.Dimension <= 0 {
		// Will be auto-detected at runtime, but warn if explicitly set to invalid value
		observability.Warnf("Vector field dimension is invalid (%d), will be auto-detected",
			config.Collection.VectorField.Dimension)
	}

	// Validate index configuration
	if config.Collection.Index.Type == "" {
		config.Collection.Index.Type = "HNSW" // Default index type
	}
	if config.Collection.Index.Params.M <= 0 {
		config.Collection.Index.Params.M = 16 // Default M parameter
	}
	if config.Collection.Index.Params.EfConstruction <= 0 {
		config.Collection.Index.Params.EfConstruction = 64 // Default efConstruction
	}

	// Validate search configuration
	if config.Search.TopK <= 0 {
		config.Search.TopK = 10 // Default topK
	}
	if config.Search.Params.Ef <= 0 {
		config.Search.Params.Ef = 64 // Default ef
	}
	if config.Search.ConsistencyLevel == "" {
		config.Search.ConsistencyLevel = "Session" // Default consistency level
	}

	// Validate performance settings
	if config.Performance.ConnectionPool.MaxConnections <= 0 {
		config.Performance.ConnectionPool.MaxConnections = 10 // Default max connections
	}
	if config.Performance.ConnectionPool.MaxIdleConnections <= 0 {
		config.Performance.ConnectionPool.MaxIdleConnections = 5 // Default idle connections
	}
	if config.Performance.ConnectionPool.AcquireTimeout <= 0 {
		config.Performance.ConnectionPool.AcquireTimeout = 5 // Default acquire timeout
	}
	if config.Performance.Batch.InsertBatchSize <= 0 {
		config.Performance.Batch.InsertBatchSize = 1000 // Default batch size
	}

	// Validate TTL settings
	if config.DataManagement.TTL.CleanupInterval <= 0 {
		config.DataManagement.TTL.CleanupInterval = 3600 // Default 1 hour
	}
	if config.DataManagement.Compaction.Interval <= 0 {
		config.DataManagement.Compaction.Interval = 86400 // Default 24 hours
	}

	observability.Debugf("Milvus configuration validated - host=%s:%d, collection=%s, index=%s",
		config.Connection.Host, config.Connection.Port, config.Collection.Name, config.Collection.Index.Type)

	return nil
}

// validateRuntimeConfig validates configuration at runtime with embedding model
func (c *MilvusCache) validateRuntimeConfig() error {
	// Test embedding generation to ensure model is available
	testEmbedding, err := candle_binding.GetEmbedding("test", 0)
	if err != nil {
		return fmt.Errorf("failed to generate test embedding: %w", err)
	}

	actualDimension := len(testEmbedding)

	// Log the detected dimension
	observability.Infof("Detected embedding dimension: %d", actualDimension)

	// Check if configured dimension matches (if explicitly set)
	if c.config.Collection.VectorField.Dimension > 0 &&
		c.config.Collection.VectorField.Dimension != actualDimension {
		observability.Warnf("Configured vector dimension (%d) differs from detected dimension (%d), using detected dimension",
			c.config.Collection.VectorField.Dimension, actualDimension)
	}

	return nil
}

// initializeCollection sets up the Milvus collection and index structures
func (c *MilvusCache) initializeCollection() error {
	ctx := context.Background()

	// Validate runtime configuration (including embedding model)
	if err := c.validateRuntimeConfig(); err != nil {
		return fmt.Errorf("runtime configuration validation failed: %w", err)
	}

	// Verify collection existence
	hasCollection, err := c.client.HasCollection(ctx, c.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}

	// Handle development mode collection reset
	if c.config.Development.DropCollectionOnStartup && hasCollection {
		if err := c.client.DropCollection(ctx, c.collectionName); err != nil {
			observability.Debugf("MilvusCache: failed to drop collection: %v", err)
			return fmt.Errorf("failed to drop collection: %w", err)
		}
		hasCollection = false
		observability.Debugf("MilvusCache: dropped existing collection '%s' for development", c.collectionName)
		observability.LogEvent("collection_dropped", map[string]interface{}{
			"backend":    "milvus",
			"collection": c.collectionName,
			"reason":     "development_mode",
		})
	}

	// Create collection if it doesn't exist
	if !hasCollection {
		if !c.config.Development.AutoCreateCollection {
			return fmt.Errorf("collection %s does not exist and auto-creation is disabled", c.collectionName)
		}

		if err := c.createCollection(); err != nil {
			observability.Debugf("MilvusCache: failed to create collection: %v", err)
			return fmt.Errorf("failed to create collection: %w", err)
		}
		observability.Debugf("MilvusCache: created new collection '%s' with dimension %d",
			c.collectionName, c.config.Collection.VectorField.Dimension)
		observability.LogEvent("collection_created", map[string]interface{}{
			"backend":    "milvus",
			"collection": c.collectionName,
			"dimension":  c.config.Collection.VectorField.Dimension,
		})
	}

	// Load collection into memory for queries
	observability.Debugf("MilvusCache: loading collection '%s' into memory", c.collectionName)
	if err := c.client.LoadCollection(ctx, c.collectionName, false); err != nil {
		observability.Debugf("MilvusCache: failed to load collection: %v", err)
		return fmt.Errorf("failed to load collection: %w", err)
	}
	observability.Debugf("MilvusCache: collection loaded successfully")

	return nil
}

// createCollection builds the Milvus collection with the appropriate schema
func (c *MilvusCache) createCollection() error {
	ctx := context.Background()

	// Determine embedding dimension automatically
	testEmbedding, err := candle_binding.GetEmbedding("test", 0) // Auto-detect
	if err != nil {
		return fmt.Errorf("failed to detect embedding dimension: %w", err)
	}
	actualDimension := len(testEmbedding)

	observability.Debugf("MilvusCache.createCollection: auto-detected embedding dimension: %d", actualDimension)

	// Define schema with auto-detected dimension
	schema := &entity.Schema{
		CollectionName: c.collectionName,
		Description:    c.config.Collection.Description,
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				TypeParams: map[string]string{"max_length": "64"},
			},
			{
				Name:       "model",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "256"},
			},
			{
				Name:       "query",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:       "request_body",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:       "response_body",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:     c.config.Collection.VectorField.Name,
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", actualDimension), // Use auto-detected dimension
				},
			},
			{
				Name:     "timestamp",
				DataType: entity.FieldTypeInt64,
			},
		},
	}

	// Create collection
	if err := c.client.CreateCollection(ctx, schema, 1); err != nil {
		return err
	}

	// Create index
	indexParams := map[string]string{
		"index_type":  c.config.Collection.Index.Type,
		"metric_type": c.config.Collection.VectorField.MetricType,
		"params": fmt.Sprintf(`{"M": %d, "efConstruction": %d}`,
			c.config.Collection.Index.Params.M,
			c.config.Collection.Index.Params.EfConstruction),
	}

	observability.Debugf("MilvusCache.createCollection: creating index for %d-dimensional vectors", actualDimension)

	// Create index with updated API
	index := entity.NewGenericIndex(c.config.Collection.VectorField.Name, entity.IndexType(c.config.Collection.Index.Type), indexParams)
	if err := c.client.CreateIndex(ctx, c.collectionName, c.config.Collection.VectorField.Name, index, false); err != nil {
		return err
	}

	return nil
}

// IsEnabled returns the current cache activation status
func (c *MilvusCache) IsEnabled() bool {
	return c.enabled
}

// AddPendingRequest stores a request that is awaiting its response
func (c *MilvusCache) AddPendingRequest(model string, query string, requestBody []byte) (string, error) {
	start := time.Now()

	if !c.enabled {
		return query, nil
	}

	// For pending entries, we need to store them immediately to ensure they can be found later
	// This is a special case that bypasses batch writing for correctness
	result, err := c.addPendingEntry(model, query, requestBody)

	if err != nil {
		metrics.RecordCacheOperation("milvus", "add_pending", "error", time.Since(start).Seconds())
	} else {
		metrics.RecordCacheOperation("milvus", "add_pending", "success", time.Since(start).Seconds())
	}

	return result, err
}

// addPendingEntry stores a pending entry that will be completed later
// This method bypasses batch writing to ensure immediate availability
func (c *MilvusCache) addPendingEntry(model string, query string, requestBody []byte) (string, error) {
	// Generate semantic embedding for the query
	embedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		return "", fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Generate unique ID
	id := fmt.Sprintf("%x", md5.Sum([]byte(fmt.Sprintf("%s_%s_%d_pending", model, query, time.Now().UnixNano()))))

	// Create context with timeout
	timeout := time.Duration(c.config.Connection.Timeout) * time.Second
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Prepare data for insertion
	ids := []string{id}
	models := []string{model}
	queries := []string{query}
	requestBodies := []string{string(requestBody)}
	responseBodies := []string{""} // Empty response body for pending entries
	embeddings := [][]float32{embedding}
	timestamps := []int64{time.Now().Unix()}

	// Create columns
	idColumn := entity.NewColumnVarChar("id", ids)
	modelColumn := entity.NewColumnVarChar("model", models)
	queryColumn := entity.NewColumnVarChar("query", queries)
	requestColumn := entity.NewColumnVarChar("request_body", requestBodies)
	responseColumn := entity.NewColumnVarChar("response_body", responseBodies)
	embeddingColumn := entity.NewColumnFloatVector(c.config.Collection.VectorField.Name, len(embedding), embeddings)
	timestampColumn := entity.NewColumnInt64("timestamp", timestamps)

	// Insert the pending entry immediately with retry logic
	observability.Debugf("MilvusCache.addPendingEntry: inserting pending entry for query '%s'", query)

	insertErr := c.retryOperation(ctx, func() error {
		_, err := c.client.Insert(ctx, c.collectionName, "", idColumn, modelColumn, queryColumn, requestColumn, responseColumn, embeddingColumn, timestampColumn)
		return err
	}, "insert pending entry")

	if insertErr != nil {
		if c.isContextError(insertErr) {
			observability.Debugf("MilvusCache.addPendingEntry: insert timeout/cancelled: %v", insertErr)
		} else {
			observability.Warnf("MilvusCache.addPendingEntry: insert failed after retries: %v", insertErr)
		}
		return "", fmt.Errorf("failed to insert pending entry: %w", insertErr)
	}

	// Flush to ensure persistence
	flushCtx, flushCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer flushCancel()

	if err := c.client.Flush(flushCtx, c.collectionName, false); err != nil {
		if !c.isContextError(err) {
			observability.Warnf("Failed to flush pending entry: %v", err)
		}
	}

	observability.Debugf("MilvusCache.addPendingEntry: successfully stored pending entry")
	return query, nil
}

// UpdateWithResponse completes a pending request by adding the response
func (c *MilvusCache) UpdateWithResponse(query string, responseBody []byte) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	queryPreview := query
	if len(query) > 50 {
		queryPreview = query[:50] + "..."
	}

	observability.Debugf("MilvusCache.UpdateWithResponse: updating pending entry (query: %s, response_size: %d)",
		queryPreview, len(responseBody))

	// Find the pending entry and complete it with the response
	// Query for the incomplete entry to retrieve its metadata
	ctx := context.Background()
	queryExpr := fmt.Sprintf("query == \"%s\" && response_body == \"\"", query)

	observability.Debugf("MilvusCache.UpdateWithResponse: searching for pending entry with expr: %s", queryExpr)

	results, err := c.client.Query(ctx, c.collectionName, []string{}, queryExpr,
		[]string{"model", "request_body"})

	if err != nil {
		observability.Debugf("MilvusCache.UpdateWithResponse: query failed: %v", err)
		metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to query pending entry: %w", err)
	}

	if len(results) == 0 {
		observability.Debugf("MilvusCache.UpdateWithResponse: no pending entry found, adding as new complete entry")
		// Create new complete entry when no pending entry exists
		// Use AddEntry to leverage batch writing if enabled
		err := c.AddEntry("unknown", query, []byte(""), responseBody)
		if err != nil {
			metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
		} else {
			metrics.RecordCacheOperation("milvus", "update_response", "success", time.Since(start).Seconds())
		}
		return err
	}

	// Get the model and request body from the pending entry
	modelColumn := results[0].(*entity.ColumnVarChar)
	requestColumn := results[1].(*entity.ColumnVarChar)

	if modelColumn.Len() > 0 {
		model := modelColumn.Data()[0]
		requestBody := requestColumn.Data()[0]

		observability.Debugf("MilvusCache.UpdateWithResponse: found pending entry, adding complete entry (model: %s)", model)

		// Create the complete entry with response data
		_, err := c.addEntry(model, query, []byte(requestBody), responseBody)
		if err != nil {
			metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
			return fmt.Errorf("failed to add complete entry: %w", err)
		}

		observability.Debugf("MilvusCache.UpdateWithResponse: successfully added complete entry with response")
		metrics.RecordCacheOperation("milvus", "update_response", "success", time.Since(start).Seconds())
	}

	return nil
}

// AddEntry stores a complete request-response pair in the cache
func (c *MilvusCache) AddEntry(model string, query string, requestBody, responseBody []byte) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	// Check if batch writing is enabled and appropriate
	if c.config.Performance.Batch.InsertBatchSize > 1 {
		// Generate embedding for batch entry
		embedding, err := candle_binding.GetEmbedding(query, 0)
		if err != nil {
			metrics.RecordCacheOperation("milvus", "add_entry", "error", time.Since(start).Seconds())
			return fmt.Errorf("failed to generate embedding: %w", err)
		}

		// Create cache entry for batch
		entry := &CacheEntry{
			RequestBody:  requestBody,
			ResponseBody: responseBody,
			Model:        model,
			Query:        query,
			Embedding:    embedding,
			Timestamp:    time.Now(),
		}

		// Add to batch buffer
		err = c.addToBatch(entry)
		if err != nil {
			metrics.RecordCacheOperation("milvus", "add_entry", "error", time.Since(start).Seconds())
			return err
		}

		c.updateWriteMetrics(time.Since(start), true)
		metrics.RecordCacheOperation("milvus", "add_entry", "success", time.Since(start).Seconds())
		return nil
	}

	// Fallback to single write
	_, err := c.addEntry(model, query, requestBody, responseBody)
	if err != nil {
		metrics.RecordCacheOperation("milvus", "add_entry", "error", time.Since(start).Seconds())
	} else {
		c.updateWriteMetrics(time.Since(start), false)
		metrics.RecordCacheOperation("milvus", "add_entry", "success", time.Since(start).Seconds())
	}

	return err
}

// addEntry handles the internal logic for storing entries in Milvus
func (c *MilvusCache) addEntry(model string, query string, requestBody, responseBody []byte) (string, error) {
	// Generate semantic embedding for the query
	embedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		return "", fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Generate unique ID
	id := fmt.Sprintf("%x", md5.Sum([]byte(fmt.Sprintf("%s_%s_%d", model, query, time.Now().UnixNano()))))

	// Create context with timeout for the insert operation
	timeout := time.Duration(c.config.Connection.Timeout) * time.Second
	if timeout <= 0 {
		timeout = 30 * time.Second // Default timeout
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Prepare data for insertion
	ids := []string{id}
	models := []string{model}
	queries := []string{query}
	requestBodies := []string{string(requestBody)}
	responseBodies := []string{string(responseBody)}
	embeddings := [][]float32{embedding}
	timestamps := []int64{time.Now().Unix()}

	// Create columns
	idColumn := entity.NewColumnVarChar("id", ids)
	modelColumn := entity.NewColumnVarChar("model", models)
	queryColumn := entity.NewColumnVarChar("query", queries)
	requestColumn := entity.NewColumnVarChar("request_body", requestBodies)
	responseColumn := entity.NewColumnVarChar("response_body", responseBodies)
	embeddingColumn := entity.NewColumnFloatVector(c.config.Collection.VectorField.Name, len(embedding), embeddings)
	timestampColumn := entity.NewColumnInt64("timestamp", timestamps)

	// Insert the entry into the collection with retry logic
	observability.Debugf("MilvusCache.addEntry: inserting entry into collection '%s' (embedding_dim: %d, request_size: %d, response_size: %d)",
		c.collectionName, len(embedding), len(requestBody), len(responseBody))

	insertErr := c.retryOperation(ctx, func() error {
		_, err := c.client.Insert(ctx, c.collectionName, "", idColumn, modelColumn, queryColumn, requestColumn, responseColumn, embeddingColumn, timestampColumn)
		return err
	}, "insert cache entry")

	if insertErr != nil {
		if c.isContextError(insertErr) {
			observability.Debugf("MilvusCache.addEntry: insert timeout/cancelled: %v", insertErr)
		} else {
			observability.Warnf("MilvusCache.addEntry: insert failed after retries: %v", insertErr)
		}
		return "", fmt.Errorf("failed to insert cache entry: %w", insertErr)
	}

	// Ensure data is persisted to storage (with shorter timeout for flush)
	flushCtx, flushCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer flushCancel()

	if err := c.client.Flush(flushCtx, c.collectionName, false); err != nil {
		if !c.isContextError(err) {
			observability.Warnf("Failed to flush cache entry: %v", err)
		}
	}

	observability.Debugf("MilvusCache.addEntry: successfully added entry to Milvus")
	observability.LogEvent("cache_entry_added", map[string]interface{}{
		"backend":             "milvus",
		"collection":          c.collectionName,
		"query":               query,
		"model":               model,
		"embedding_dimension": len(embedding),
	})
	return query, nil
}

// FindSimilar searches for semantically similar cached requests using native vector search
func (c *MilvusCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	start := time.Now()

	if !c.enabled {
		observability.Debugf("MilvusCache.FindSimilar: cache disabled")
		return nil, false, nil
	}
	queryPreview := query
	if len(query) > 50 {
		queryPreview = query[:50] + "..."
	}
	observability.Debugf("MilvusCache.FindSimilar: searching for model='%s', query='%s' (len=%d chars)",
		model, queryPreview, len(query))

	// Generate semantic embedding for similarity comparison
	queryEmbedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Create context with timeout for the search operation
	timeout := time.Duration(c.config.Connection.Timeout) * time.Second
	if timeout <= 0 {
		timeout = 30 * time.Second // Default timeout
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Create vector search parameters
	searchParam, err := entity.NewIndexHNSWSearchParam(c.config.Search.Params.Ef)
	if err != nil {
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, fmt.Errorf("failed to create search param: %w", err)
	}

	// Create search vector
	searchVector := entity.FloatVector(queryEmbedding)

	// Set up output fields to retrieve
	outputFields := []string{"response_body", "query", "model"}

	// Create filter expression for the model and non-empty responses
	filterExpr := fmt.Sprintf("model == \"%s\" && response_body != \"\"", model)
	observability.Debugf("MilvusCache.FindSimilar: vector search with expr: %s, topk=%d, ef=%d (embedding_dim: %d)",
		filterExpr, c.config.Search.TopK, c.config.Search.Params.Ef, len(queryEmbedding))

	// Perform vector search with retry logic
	var searchResults []client.SearchResult
	var searchErr error

	searchErr = c.retryOperation(ctx, func() error {
		results, err := c.client.Search(
			ctx,
			c.collectionName,
			[]string{}, // partition names (empty means search all partitions)
			filterExpr,
			outputFields,
			[]entity.Vector{searchVector},
			c.config.Collection.VectorField.Name,
			entity.L2, // metric type
			c.config.Search.TopK,
			searchParam,
		)
		if err != nil {
			return err
		}
		searchResults = results
		return nil
	}, "vector search")

	if searchErr != nil {
		if c.isContextError(searchErr) {
			observability.Debugf("MilvusCache.FindSimilar: search timeout/cancelled: %v", searchErr)
		} else {
			observability.Warnf("MilvusCache.FindSimilar: vector search failed after retries: %v", searchErr)
		}
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, fmt.Errorf("vector search failed: %w", searchErr)
	}

	// Check if we got any results
	if len(searchResults) == 0 || searchResults[0].IDs.Len() == 0 {
		atomic.AddInt64(&c.missCount, 1)
		observability.Debugf("MilvusCache.FindSimilar: no similar entries found via vector search")
		metrics.RecordCacheOperation("milvus", "find_similar", "miss", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	// Process search results
	bestSimilarity := float32(-1.0)
	var bestResponse string
	entriesChecked := 0

	// Iterate through search results to find the best match above threshold
	for _, result := range searchResults {
		for i := 0; i < result.IDs.Len(); i++ {
			entriesChecked++

			// Get similarity score (distance) - for IP metric, higher is better
			similarity := float32(0.0)
			if result.Scores != nil && i < len(result.Scores) {
				// For IP (Inner Product) metric, scores are already similarity values
				similarity = result.Scores[i]
			}

			// Get response body
			var responseBody string
			for _, field := range result.Fields {
				if field.Name() == "response_body" {
					if column, ok := field.(*entity.ColumnVarChar); ok {
						responseBody = column.Data()[i]
					}
					break
				}
			}

			observability.Debugf("MilvusCache.FindSimilar: result %d - similarity=%.4f, response_size=%d",
				i+1, similarity, len(responseBody))

			// Update best match if this result has higher similarity
			if similarity > bestSimilarity && similarity >= c.similarityThreshold {
				bestSimilarity = similarity
				bestResponse = responseBody
			}
		}
	}

	observability.Debugf("MilvusCache.FindSimilar: best similarity=%.4f, threshold=%.4f (checked %d entries)",
		bestSimilarity, c.similarityThreshold, entriesChecked)

	if bestSimilarity >= c.similarityThreshold && bestResponse != "" {
		atomic.AddInt64(&c.hitCount, 1)
		observability.Debugf("MilvusCache.FindSimilar: CACHE HIT - similarity=%.4f >= threshold=%.4f, response_size=%d bytes",
			bestSimilarity, c.similarityThreshold, len(bestResponse))
		observability.LogEvent("cache_hit", map[string]interface{}{
			"backend":       "milvus",
			"similarity":    bestSimilarity,
			"threshold":     c.similarityThreshold,
			"model":         model,
			"collection":    c.collectionName,
			"search_method": "vector_search",
			"entries_found": entriesChecked,
		})
		metrics.RecordCacheOperation("milvus", "find_similar", "hit", time.Since(start).Seconds())
		metrics.RecordCacheHit()
		return []byte(bestResponse), true, nil
	}

	atomic.AddInt64(&c.missCount, 1)
	observability.Debugf("MilvusCache.FindSimilar: CACHE MISS - best_similarity=%.4f < threshold=%.4f",
		bestSimilarity, c.similarityThreshold)
	observability.LogEvent("cache_miss", map[string]interface{}{
		"backend":         "milvus",
		"best_similarity": bestSimilarity,
		"threshold":       c.similarityThreshold,
		"model":           model,
		"collection":      c.collectionName,
		"search_method":   "vector_search",
		"entries_checked": entriesChecked,
	})
	metrics.RecordCacheOperation("milvus", "find_similar", "miss", time.Since(start).Seconds())
	metrics.RecordCacheMiss()
	return nil, false, nil
}

// Close releases all resources held by the cache
func (c *MilvusCache) Close() error {
	// Stop TTL cleanup goroutine if running
	c.stopTTLCleanup()

	// Close Milvus client
	if c.client != nil {
		observability.Debugf("MilvusCache: closing Milvus client connection")
		err := c.client.Close()
		if err != nil {
			observability.Warnf("Failed to close Milvus client: %v", err)
			return err
		}
		observability.Debugf("MilvusCache: Milvus client connection closed")
	}
	return nil
}

// GetStats provides current cache performance metrics
func (c *MilvusCache) GetStats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	hits := atomic.LoadInt64(&c.hitCount)
	misses := atomic.LoadInt64(&c.missCount)
	total := hits + misses

	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	// Retrieve collection statistics from Milvus
	totalEntries := 0
	if c.enabled && c.client != nil {
		ctx := context.Background()
		stats, err := c.client.GetCollectionStatistics(ctx, c.collectionName)
		if err == nil {
			// Extract entity count from statistics
			if entityCount, ok := stats["row_count"]; ok {
				fmt.Sscanf(entityCount, "%d", &totalEntries)
				observability.Debugf("MilvusCache.GetStats: collection '%s' contains %d entries",
					c.collectionName, totalEntries)
			}
		} else {
			observability.Debugf("MilvusCache.GetStats: failed to get collection stats: %v", err)
		}
	}

	cacheStats := CacheStats{
		TotalEntries: totalEntries,
		HitCount:     hits,
		MissCount:    misses,
		HitRatio:     hitRatio,
	}

	if c.lastCleanupTime != nil {
		cacheStats.LastCleanupTime = c.lastCleanupTime
	}

	return cacheStats
}

// startTTLCleanup starts a background goroutine to clean up expired entries
func (c *MilvusCache) startTTLCleanup() {
	c.cleanupWG.Add(1)
	go func() {
		defer c.cleanupWG.Done()

		// Calculate ticker interval from config
		interval := time.Duration(c.config.DataManagement.TTL.CleanupInterval) * time.Second
		if interval <= 0 {
			interval = time.Hour // Default to 1 hour
		}

		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		observability.Infof("MilvusCache TTL cleanup started - interval: %v, TTL: %ds", interval, c.ttlSeconds)

		for {
			select {
			case <-ticker.C:
				c.performTTLCleanup()
			case <-c.cleanupStopChan:
				observability.Infof("MilvusCache TTL cleanup stopped")
				return
			}
		}
	}()
}

// performTTLCleanup removes expired entries from the cache
func (c *MilvusCache) performTTLCleanup() {
	if c.ttlSeconds <= 0 {
		return // TTL not configured
	}

	ctx := context.Background()
	start := time.Now()

	// Calculate cutoff time
	cutoffTime := time.Now().Add(-time.Duration(c.ttlSeconds) * time.Second).Unix()
	observability.Debugf("MilvusCache.performTTLCleanup: removing entries older than %d (timestamp < %d)",
		c.ttlSeconds, cutoffTime)

	// Create expression to delete expired entries
	deleteExpr := fmt.Sprintf("timestamp < %d", cutoffTime)

	// Execute deletion
	err := c.client.Delete(ctx, c.collectionName, "", deleteExpr)
	if err != nil {
		observability.Warnf("MilvusCache TTL cleanup failed: %v", err)
		return
	}

	// Update last cleanup time
	cleanupTime := time.Now()
	c.mu.Lock()
	c.lastCleanupTime = &cleanupTime
	c.mu.Unlock()

	observability.Infof("MilvusCache TTL cleanup completed in %v", time.Since(start))
	observability.LogEvent("ttl_cleanup_completed", map[string]interface{}{
		"backend":          "milvus",
		"cleanup_time":     time.Since(start).String(),
		"ttl_seconds":      c.ttlSeconds,
		"cutoff_timestamp": cutoffTime,
	})

	// Flush changes to ensure persistence
	if err := c.client.Flush(ctx, c.collectionName, false); err != nil {
		observability.Warnf("Failed to flush after TTL cleanup: %v", err)
	}
}

// stopTTLCleanup gracefully stops the TTL cleanup and batch writer goroutines
func (c *MilvusCache) stopTTLCleanup() {
	if c.cleanupStopChan != nil {
		observability.Debugf("MilvusCache: stopping background goroutines")
		close(c.cleanupStopChan)
		c.cleanupWG.Wait()
		observability.Debugf("MilvusCache: background goroutines stopped")
	}
}

// retryOperation executes a function with retry logic for transient errors
func (c *MilvusCache) retryOperation(ctx context.Context, operation func() error, operationName string) error {
	const maxRetries = 3
	const baseDelay = 100 * time.Millisecond

	var lastErr error

	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			// Check context before retry
			if ctx.Err() != nil {
				return fmt.Errorf("context cancelled during %s: %w", operationName, ctx.Err())
			}

			// Exponential backoff
			delay := baseDelay * time.Duration(1<<uint(attempt-1))
			observability.Debugf("MilvusCache: retrying %s (attempt %d/%d) after %v",
				operationName, attempt+1, maxRetries, delay)

			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return fmt.Errorf("context cancelled during %s retry delay: %w", operationName, ctx.Err())
			}
		}

		err := operation()
		if err == nil {
			return nil // Success
		}

		lastErr = err

		// Don't retry on certain errors
		if c.isNonRetryableError(err) {
			observability.Warnf("MilvusCache: non-retryable error in %s: %v", operationName, err)
			return err
		}

		observability.Warnf("MilvusCache: %s failed (attempt %d/%d): %v",
			operationName, attempt+1, maxRetries, err)
	}

	return fmt.Errorf("%s failed after %d attempts, last error: %w", operationName, maxRetries, lastErr)
}

// isNonRetryableError determines if an error should not be retried
func (c *MilvusCache) isNonRetryableError(err error) bool {
	errStr := err.Error()

	// Don't retry authentication errors
	if strings.Contains(errStr, "authentication") ||
		strings.Contains(errStr, "unauthorized") ||
		strings.Contains(errStr, "access denied") {
		return true
	}

	// Don't retry collection not found errors
	if strings.Contains(errStr, "collection not found") ||
		strings.Contains(errStr, "does not exist") {
		return true
	}

	// Don't retry invalid configuration errors
	if strings.Contains(errStr, "invalid configuration") ||
		strings.Contains(errStr, "invalid parameter") ||
		strings.Contains(errStr, "dimension mismatch") {
		return true
	}

	return false
}

// isContextError checks if the error is context-related
func (c *MilvusCache) isContextError(err error) bool {
	return errors.Is(err, context.Canceled) ||
		errors.Is(err, context.DeadlineExceeded) ||
		strings.Contains(err.Error(), "context canceled") ||
		strings.Contains(err.Error(), "deadline exceeded")
}

// startBatchWriter starts the batch write goroutine
func (c *MilvusCache) startBatchWriter() {
	c.batchTimer = time.NewTimer(time.Duration(c.config.Performance.Batch.Timeout) * time.Second)

	go func() {
		for {
			select {
			case <-c.batchTimer.C:
				c.flushBatch()
			case <-c.cleanupStopChan:
				c.flushBatch() // Flush remaining entries before stopping
				if c.batchTimer != nil {
					c.batchTimer.Stop()
				}
				return
			}
		}
	}()

	observability.Debugf("MilvusCache: batch writer started with timeout %ds", c.config.Performance.Batch.Timeout)
}

// addToBatch adds an entry to the batch buffer
func (c *MilvusCache) addToBatch(entry *CacheEntry) error {
	c.batchMu.Lock()
	defer c.batchMu.Unlock()

	c.batchBuffer = append(c.batchBuffer, entry)

	// If batch is full, flush immediately
	if len(c.batchBuffer) >= c.config.Performance.Batch.InsertBatchSize {
		go c.flushBatch()
	} else if c.batchTimer != nil {
		// Reset timer to extend the batch window
		c.batchTimer.Reset(time.Duration(c.config.Performance.Batch.Timeout) * time.Second)
	}

	return nil
}

// flushBatch writes all buffered entries to Milvus
func (c *MilvusCache) flushBatch() {
	c.batchMu.Lock()
	if len(c.batchBuffer) == 0 {
		c.batchMu.Unlock()
		return
	}

	// Copy buffer and clear it
	entries := make([]*CacheEntry, len(c.batchBuffer))
	copy(entries, c.batchBuffer)
	c.batchBuffer = c.batchBuffer[:0]
	c.batchMu.Unlock()

	if len(entries) == 0 {
		return
	}

	start := time.Now()
	observability.Debugf("MilvusCache: flushing batch of %d entries", len(entries))

	// Prepare batch data
	ids := make([]string, len(entries))
	models := make([]string, len(entries))
	queries := make([]string, len(entries))
	requestBodies := make([]string, len(entries))
	responseBodies := make([]string, len(entries))
	embeddings := make([][]float32, len(entries))
	timestamps := make([]int64, len(entries))

	for i, entry := range entries {
		ids[i] = fmt.Sprintf("%x", md5.Sum([]byte(fmt.Sprintf("%s_%s_%d_%d",
			entry.Model, entry.Query, entry.Timestamp.UnixNano(), i))))
		models[i] = entry.Model
		queries[i] = entry.Query
		requestBodies[i] = string(entry.RequestBody)
		responseBodies[i] = string(entry.ResponseBody)
		embeddings[i] = entry.Embedding
		timestamps[i] = entry.Timestamp.Unix()
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Create columns
	idColumn := entity.NewColumnVarChar("id", ids)
	modelColumn := entity.NewColumnVarChar("model", models)
	queryColumn := entity.NewColumnVarChar("query", queries)
	requestColumn := entity.NewColumnVarChar("request_body", requestBodies)
	responseColumn := entity.NewColumnVarChar("response_body", responseBodies)
	embeddingColumn := entity.NewColumnFloatVector(c.config.Collection.VectorField.Name, len(embeddings[0]), embeddings)
	timestampColumn := entity.NewColumnInt64("timestamp", timestamps)

	// Insert batch with retry logic
	err := c.retryOperation(ctx, func() error {
		_, err := c.client.Insert(ctx, c.collectionName, "",
			idColumn, modelColumn, queryColumn, requestColumn, responseColumn, embeddingColumn, timestampColumn)
		return err
	}, "batch insert")

	// Update metrics
	c.mu.Lock()
	defer c.mu.Unlock()

	c.writeMetrics.totalWrites += int64(len(entries))

	if err != nil {
		c.writeMetrics.writeErrors += int64(len(entries))
		observability.Warnf("MilvusCache: batch insert failed for %d entries: %v", len(entries), err)
	} else {
		c.writeMetrics.batchWrites += int64(len(entries))
		c.writeMetrics.avgWriteLatency = (c.writeMetrics.avgWriteLatency*float64(c.writeMetrics.batchWrites-1) +
			time.Since(start).Seconds()) / float64(c.writeMetrics.batchWrites)
		c.writeMetrics.lastWriteTime = time.Now()

		observability.Debugf("MilvusCache: batch insert completed - %d entries in %v",
			len(entries), time.Since(start))
		observability.LogEvent("batch_insert_completed", map[string]interface{}{
			"backend":     "milvus",
			"batch_size":  len(entries),
			"duration":    time.Since(start).String(),
			"avg_latency": time.Since(start).Seconds() / float64(len(entries)),
		})

		// Flush to ensure persistence
		if flushErr := c.client.Flush(ctx, c.collectionName, false); flushErr != nil {
			observability.Warnf("Failed to flush after batch insert: %v", flushErr)
		}
	}
}

// getPerformanceMetrics returns performance metrics for the cache
func (c *MilvusCache) getPerformanceMetrics() map[string]interface{} {
	c.mu.Lock()
	defer c.mu.Unlock()

	return map[string]interface{}{
		"total_writes":      c.writeMetrics.totalWrites,
		"batch_writes":      c.writeMetrics.batchWrites,
		"single_writes":     c.writeMetrics.singleWrites,
		"write_errors":      c.writeMetrics.writeErrors,
		"avg_write_latency": c.writeMetrics.avgWriteLatency,
		"last_write_time":   c.writeMetrics.lastWriteTime,
		"batch_buffer_size": len(c.batchBuffer),
		"max_batch_size":    c.config.Performance.Batch.InsertBatchSize,
	}
}

// updateWriteMetrics updates write performance metrics
func (c *MilvusCache) updateWriteMetrics(duration time.Duration, isBatch bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.writeMetrics.totalWrites++
	if isBatch {
		c.writeMetrics.batchWrites++
	} else {
		c.writeMetrics.singleWrites++
	}

	c.writeMetrics.avgWriteLatency = (c.writeMetrics.avgWriteLatency*float64(c.writeMetrics.totalWrites-1) +
		duration.Seconds()) / float64(c.writeMetrics.totalWrites)
	c.writeMetrics.lastWriteTime = time.Now()
}
