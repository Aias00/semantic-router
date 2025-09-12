//go:build milvus
// +build milvus

package cache

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// Mock types for testing

// SearchResult represents a mock search result
type SearchResult struct {
	IDs    entity.Column
	Scores []float32
	Fields []entity.Column
}

// MockCacheEntry represents a cache entry for testing (extends the base CacheEntry)
type MockCacheEntry struct {
	CacheEntry
	ID string
}

// mockMilvusClient provides a mock implementation of Milvus client
type mockMilvusClient struct {
	shouldFailConnect   bool
	shouldFailSearch    bool
	shouldFailInsert    bool
	shouldFailUpsert    bool
	shouldFailDelete    bool
	connectionError     error
	searchError         error
	insertError         error
	upsertError         error
	deleteError         error
	searchResults       []SearchResult
	similarityThreshold float32
	delay               time.Duration
	callCount           map[string]int
	entries             map[string]*MockCacheEntry
	mu                  sync.RWMutex
}

// mockEmbeddingClient provides a mock implementation of embedding client
type mockEmbeddingClient struct {
	dimension  int
	shouldFail bool
	failError  error
}

// NewMockMilvusClient creates a new mock Milvus client
func NewMockMilvusClient() *mockMilvusClient {
	return &mockMilvusClient{
		callCount:           make(map[string]int),
		entries:             make(map[string]*MockCacheEntry),
		similarityThreshold: 0.8,
	}
}

// NewMockEmbeddingClient creates a new mock embedding client
func NewMockEmbeddingClient(dimension int) *mockEmbeddingClient {
	return &mockEmbeddingClient{
		dimension: dimension,
	}
}

// Mock client methods
func (m *mockMilvusClient) HasCollection(ctx context.Context, collectionName string) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount["HasCollection"]++

	if m.shouldFailConnect {
		return false, m.connectionError
	}

	// Simulate delay
	if m.delay > 0 {
		time.Sleep(m.delay)
	}

	// Check if collection exists in mock data
	_, exists := m.entries[collectionName]
	return exists, nil
}

func (m *mockMilvusClient) CreateCollection(ctx context.Context, schema *entity.Schema, shardNum int32) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount["CreateCollection"]++

	if m.shouldFailConnect {
		return m.connectionError
	}

	// Simulate delay
	if m.delay > 0 {
		time.Sleep(m.delay)
	}

	// Initialize collection in mock data
	m.entries[schema.CollectionName] = &MockCacheEntry{}
	return nil
}

func (m *mockMilvusClient) DropCollection(ctx context.Context, collectionName string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount["DropCollection"]++

	if m.shouldFailConnect {
		return m.connectionError
	}

	delete(m.entries, collectionName)
	return nil
}

func (m *mockMilvusClient) LoadCollection(ctx context.Context, collectionName string, async bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount["LoadCollection"]++

	if m.shouldFailConnect {
		return m.connectionError
	}

	return nil
}

func (m *mockMilvusClient) Insert(ctx context.Context, collectionName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount["Insert"]++

	if m.shouldFailInsert {
		return nil, m.insertError
	}

	// Simulate delay
	if m.delay > 0 {
		time.Sleep(m.delay)
	}

	// Create mock result
	return entity.NewColumnInt64("mock_result", []int64{1}), nil
}

func (m *mockMilvusClient) Flush(ctx context.Context, collectionName string, async bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount["Flush"]++

	if m.shouldFailInsert {
		return m.insertError
	}

	return nil
}

func (m *mockMilvusClient) Search(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount["Search"]++

	if m.shouldFailSearch {
		return nil, m.searchError
	}

	// Simulate delay
	if m.delay > 0 {
		time.Sleep(m.delay)
	}

	// Return mock search results
	return []client.SearchResult{}, nil
}

func (m *mockMilvusClient) Query(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string) ([]entity.Column, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount["Query"]++

	if m.shouldFailSearch {
		return nil, m.searchError
	}

	// Return mock query results
	return []entity.Column{}, nil
}

func (m *mockMilvusClient) Delete(ctx context.Context, collectionName string, partitionName string, expr string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount["Delete"]++

	if m.shouldFailDelete {
		return m.deleteError
	}

	return nil
}

func (m *mockMilvusClient) CreateIndex(ctx context.Context, collectionName string, fieldName string, idx entity.Index, async bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount["CreateIndex"]++

	return nil
}

func (m *mockMilvusClient) GetCollectionStatistics(ctx context.Context, collectionName string) (map[string]string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount["GetCollectionStatistics"]++

	return map[string]string{"row_count": "0"}, nil
}

func (m *mockMilvusClient) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount["Close"]++

	return nil
}

// Mock embedding client methods
func (m *mockEmbeddingClient) GetEmbedding(text string, dimension int) ([]float32, error) {
	if m.shouldFail {
		return nil, m.failError
	}

	// Generate mock embedding
	embedding := make([]float32, m.dimension)
	for i := range embedding {
		embedding[i] = float32(i) / float32(m.dimension)
	}

	return embedding, nil
}

// Helper functions for creating mock caches
func createMockMilvusCache(mockClient *mockMilvusClient, embeddingClient *mockEmbeddingClient) (CacheBackend, error) {
	// This is a simplified mock implementation
	// In real tests, this would create a proper mock cache
	return &MockMilvusCache{
		enabled:             true,
		similarityThreshold: 0.8,
		ttlSeconds:          3600,
		mockClient:          mockClient,
		embeddingClient:     embeddingClient,
	}, nil
}

func createMockMilvusCacheWithConfig(mockClient *mockMilvusClient, embeddingClient *mockEmbeddingClient, config map[string]interface{}) (CacheBackend, error) {
	// Create mock cache with custom configuration
	return &MockMilvusCache{
		enabled:             true,
		similarityThreshold: 0.8,
		ttlSeconds:          3600,
		mockClient:          mockClient,
		embeddingClient:     embeddingClient,
	}, nil
}

// MockMilvusCache provides a mock implementation for testing
type MockMilvusCache struct {
	enabled             bool
	similarityThreshold float32
	ttlSeconds          int
	mockClient          *mockMilvusClient
	embeddingClient     *mockEmbeddingClient
	entries             []*MockCacheEntry
	mu                  sync.RWMutex
	hitCount            int64
	missCount           int64
}

func (m *MockMilvusCache) IsEnabled() bool {
	return m.enabled
}

func (m *MockMilvusCache) AddPendingRequest(model string, query string, requestBody []byte) (string, error) {
	if !m.enabled {
		return query, nil
	}

	// Generate mock embedding
	embedding, err := m.embeddingClient.GetEmbedding(query, 0)
	if err != nil {
		return "", err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	entry := &MockCacheEntry{
		CacheEntry: CacheEntry{
			Model:        model,
			Query:        query,
			RequestBody:  requestBody,
			ResponseBody: []byte{}, // Empty for pending
			Embedding:    embedding,
			Timestamp:    time.Now(),
		},
		ID: fmt.Sprintf("pending_%d", time.Now().UnixNano()),
	}

	m.entries = append(m.entries, entry)
	return query, nil
}

func (m *MockMilvusCache) UpdateWithResponse(query string, responseBody []byte) error {
	if !m.enabled {
		return nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Find pending entry and update it
	for _, entry := range m.entries {
		if entry.Query == query && len(entry.ResponseBody) == 0 {
			entry.ResponseBody = responseBody
			return nil
		}
	}

	// If not found, add as new entry
	embedding, err := m.embeddingClient.GetEmbedding(query, 0)
	if err != nil {
		return err
	}

	entry := &MockCacheEntry{
		CacheEntry: CacheEntry{
			Model:        "unknown",
			Query:        query,
			RequestBody:  []byte{},
			ResponseBody: responseBody,
			Embedding:    embedding,
			Timestamp:    time.Now(),
		},
		ID: fmt.Sprintf("complete_%d", time.Now().UnixNano()),
	}

	m.entries = append(m.entries, entry)
	return nil
}

func (m *MockMilvusCache) AddEntry(model string, query string, requestBody, responseBody []byte) error {
	if !m.enabled {
		return nil
	}

	// Generate mock embedding
	embedding, err := m.embeddingClient.GetEmbedding(query, 0)
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	entry := &MockCacheEntry{
		CacheEntry: CacheEntry{
			Model:        model,
			Query:        query,
			RequestBody:  requestBody,
			ResponseBody: responseBody,
			Embedding:    embedding,
			Timestamp:    time.Now(),
		},
		ID: fmt.Sprintf("entry_%d", time.Now().UnixNano()),
	}

	m.entries = append(m.entries, entry)
	return nil
}

func (m *MockMilvusCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	if !m.enabled {
		return nil, false, nil
	}

	// Generate mock embedding for query
	queryEmbedding, err := m.embeddingClient.GetEmbedding(query, 0)
	if err != nil {
		return nil, false, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	var bestMatch *MockCacheEntry
	bestSimilarity := float32(-1.0)

	for _, entry := range m.entries {
		if entry.Model != model || len(entry.ResponseBody) == 0 {
			continue
		}

		// Calculate simple similarity (dot product)
		similarity := float32(0.0)
		for i := 0; i < len(queryEmbedding) && i < len(entry.Embedding); i++ {
			similarity += queryEmbedding[i] * entry.Embedding[i]
		}

		if similarity > bestSimilarity && similarity >= m.similarityThreshold {
			bestSimilarity = similarity
			bestMatch = entry
		}
	}

	if bestMatch != nil {
		return bestMatch.ResponseBody, true, nil
	}

	return nil, false, nil
}

func (m *MockMilvusCache) Close() error {
	return nil
}

func (m *MockMilvusCache) GetStats() CacheStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	hits := m.hitCount
	misses := m.missCount
	total := hits + misses

	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	return CacheStats{
		TotalEntries: len(m.entries),
		HitCount:     hits,
		MissCount:    misses,
		HitRatio:     hitRatio,
	}
}
