//go:build milvus
// +build milvus

package cache

import (
	"testing"
)

func TestMilvusCacheBasic(t *testing.T) {
	// Test basic functionality without complex mocks
	options := MilvusCacheOptions{
		Enabled: false, // Use disabled mode for basic test
	}

	cache, err := NewMilvusCache(options)
	if err != nil {
		t.Fatalf("Failed to create Milvus cache: %v", err)
	}

	if cache.IsEnabled() {
		t.Error("Expected cache to be disabled")
	}

	// Test basic operations on disabled cache
	err = cache.AddEntry("test", "query", []byte("req"), []byte("resp"))
	if err != nil {
		t.Errorf("AddEntry should not fail on disabled cache: %v", err)
	}

	_, found, err := cache.FindSimilar("test", "query")
	if err != nil {
		t.Errorf("FindSimilar should not fail on disabled cache: %v", err)
	}
	if found {
		t.Error("FindSimilar should not find anything on disabled cache")
	}

	err = cache.Close()
	if err != nil {
		t.Errorf("Close should not fail: %v", err)
	}
}

func TestLoadMilvusConfig(t *testing.T) {
	// Test that LoadMilvusConfig function exists and can be called
	// This will fail without a real config file, but we can test the function exists
	_, err := LoadMilvusConfig("nonexistent-config.yaml")
	if err == nil {
		t.Error("Expected error for nonexistent config file")
	}

	// Check that the error message contains expected text
	if err.Error() != "failed to read config file: open nonexistent-config.yaml: no such file or directory" {
		t.Errorf("Unexpected error message: %v", err)
	}
}
