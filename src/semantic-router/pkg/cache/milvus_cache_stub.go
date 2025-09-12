//go:build !milvus

package cache

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// MilvusCacheStub provides a stub implementation when Milvus build tag is not enabled
type MilvusCacheStub struct {
	enabled             bool
	similarityThreshold float32
	hitCount            int64
	missCount           int64
	mu                  sync.RWMutex
}

// MilvusCacheOptions contains configuration parameters for Milvus cache initialization
type MilvusCacheOptions struct {
	SimilarityThreshold float32
	TTLSeconds          int
	Enabled             bool
	ConfigPath          string
}

// NewMilvusCache creates a stub implementation when Milvus is not built
func NewMilvusCache(options MilvusCacheOptions) (*MilvusCacheStub, error) {
	if options.Enabled {
		observability.Warnf("Milvus cache was requested but was not built with milvus build tag. Using disabled stub.")
		return &MilvusCacheStub{
			enabled:             false,
			similarityThreshold: options.SimilarityThreshold,
		}, nil
	}
	
	return &MilvusCacheStub{
		enabled:             false,
		similarityThreshold: options.SimilarityThreshold,
	}, nil
}

// IsEnabled returns the current cache activation status
func (c *MilvusCacheStub) IsEnabled() bool {
	return c.enabled
}

// AddPendingRequest stores a request that is awaiting its response
func (c *MilvusCacheStub) AddPendingRequest(model string, query string, requestBody []byte) (string, error) {
	start := time.Now()
	metrics.RecordCacheOperation("milvus", "add_pending", "error", time.Since(start).Seconds())
	return query, fmt.Errorf("Milvus cache is not available - build with milvus tag to enable")
}

// UpdateWithResponse completes a pending request by adding the response
func (c *MilvusCacheStub) UpdateWithResponse(query string, responseBody []byte) error {
	start := time.Now()
	metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
	return fmt.Errorf("Milvus cache is not available - build with milvus tag to enable")
}

// AddEntry stores a complete request-response pair in the cache
func (c *MilvusCacheStub) AddEntry(model string, query string, requestBody, responseBody []byte) error {
	start := time.Now()
	metrics.RecordCacheOperation("milvus", "add_entry", "error", time.Since(start).Seconds())
	return fmt.Errorf("Milvus cache is not available - build with milvus tag to enable")
}

// FindSimilar searches for semantically similar cached requests
func (c *MilvusCacheStub) FindSimilar(model string, query string) ([]byte, bool, error) {
	start := time.Now()
	atomic.AddInt64(&c.missCount, 1)
	metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
	metrics.RecordCacheMiss()
	return nil, false, fmt.Errorf("Milvus cache is not available - build with milvus tag to enable")
}

// Close releases all resources held by the cache
func (c *MilvusCacheStub) Close() error {
	return nil
}

// performTTLCleanup stub implementation for non-milvus builds
func (c *MilvusCacheStub) performTTLCleanup() {
	// No-op for stub implementation
}

// GetStats provides current cache performance metrics
func (c *MilvusCacheStub) GetStats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	hits := atomic.LoadInt64(&c.hitCount)
	misses := atomic.LoadInt64(&c.missCount)
	total := hits + misses

	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	return CacheStats{
		TotalEntries: 0,
		HitCount:     hits,
		MissCount:    misses,
		HitRatio:     hitRatio,
	}
}