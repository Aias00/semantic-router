//go:build milvus
// +build milvus

package cache_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Milvus Configuration", func() {
	var tempDir string

	BeforeEach(func() {
		var err error
		tempDir, err = os.MkdirTemp("", "milvus-config-test")
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		if tempDir != "" {
			os.RemoveAll(tempDir)
		}
	})

	Describe("Configuration Loading", func() {
		It("should load valid configuration", func() {
			configPath := filepath.Join(tempDir, "valid-config.yaml")
			createValidConfig(configPath)

			config, err := cache.LoadMilvusConfig(configPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(config).NotTo(BeNil())

			Expect(config.Connection.Host).To(Equal("localhost"))
			Expect(config.Connection.Port).To(Equal(19530))
			Expect(config.Collection.Name).To(Equal("test_collection"))
			Expect(config.Collection.VectorField.Name).To(Equal("embedding"))
			Expect(config.Search.TopK).To(Equal(10))
		})

		It("should handle missing config file", func() {
			missingPath := filepath.Join(tempDir, "missing-config.yaml")

			_, err := cache.LoadMilvusConfig(missingPath)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("no such file or directory"))
		})

		It("should reject empty config path", func() {
			_, err := cache.LoadMilvusConfig("")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("config path is required"))
		})

		It("should handle invalid YAML", func() {
			invalidYAMLPath := filepath.Join(tempDir, "invalid-yaml.yaml")
			createInvalidYAMLConfig(invalidYAMLPath)

			_, err := cache.LoadMilvusConfig(invalidYAMLPath)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to parse config file"))
		})
	})

	Describe("Configuration Validation", func() {
		It("should validate required fields", func() {
			configPath := filepath.Join(tempDir, "invalid-fields.yaml")
			createConfigWithInvalidFields(configPath)

			_, err := cache.LoadMilvusConfig(configPath)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("configuration validation failed"))
		})

		It("should set sensible defaults", func() {
			configPath := filepath.Join(tempDir, "minimal-config.yaml")
			createMinimalConfig(configPath)

			config, err := cache.LoadMilvusConfig(configPath)
			Expect(err).NotTo(HaveOccurred())

			Expect(config.Collection.Name).To(Equal("semantic_cache"))
			Expect(config.Collection.VectorField.Name).To(Equal("embedding"))
			Expect(config.Collection.Index.Type).To(Equal("HNSW"))
			Expect(config.Search.TopK).To(Equal(10))
			Expect(config.Search.Params.Ef).To(Equal(64))
			Expect(config.Search.ConsistencyLevel).To(Equal("Session"))
			Expect(config.Performance.ConnectionPool.MaxConnections).To(Equal(10))
			Expect(config.DataManagement.TTL.CleanupInterval).To(Equal(3600))
		})

		It("should validate authentication requirements", func() {
			configPath := filepath.Join(tempDir, "auth-config.yaml")
			createAuthConfigWithoutCredentials(configPath)

			_, err := cache.LoadMilvusConfig(configPath)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("username is required when authentication is enabled"))
		})

		It("should validate TLS requirements", func() {
			configPath := filepath.Join(tempDir, "tls-config.yaml")
			createTLSConfigWithoutCert(configPath)

			_, err := cache.LoadMilvusConfig(configPath)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("both cert_file and key_file are required when TLS is enabled"))
		})

		It("should validate port range", func() {
			configPath := filepath.Join(tempDir, "invalid-port.yaml")
			createConfigWithInvalidPort(configPath)

			_, err := cache.LoadMilvusConfig(configPath)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("port must be between 1 and 65535"))
		})
	})

	Describe("Configuration Features", func() {
		It("should support authentication configuration", func() {
			configPath := filepath.Join(tempDir, "auth-config-valid.yaml")
			createValidAuthConfig(configPath)

			config, err := cache.LoadMilvusConfig(configPath)
			Expect(err).NotTo(HaveOccurred())

			Expect(config.Connection.Auth.Enabled).To(BeTrue())
			Expect(config.Connection.Auth.Username).To(Equal("test_user"))
			Expect(config.Connection.Auth.Password).To(Equal("test_password"))
		})

		It("should support TLS configuration", func() {
			configPath := filepath.Join(tempDir, "tls-config-valid.yaml")
			createValidTLSConfig(configPath)

			config, err := cache.LoadMilvusConfig(configPath)
			Expect(err).NotTo(HaveOccurred())

			Expect(config.Connection.TLS.Enabled).To(BeTrue())
			Expect(config.Connection.TLS.CertFile).To(Equal("/path/to/cert.pem"))
			Expect(config.Connection.TLS.KeyFile).To(Equal("/path/to/key.pem"))
		})

		It("should support different index types", func() {
			configPath := filepath.Join(tempDir, "ivf-config.yaml")
			createIVFFlatConfig(configPath)

			config, err := cache.LoadMilvusConfig(configPath)
			Expect(err).NotTo(HaveOccurred())

			Expect(config.Collection.Index.Type).To(Equal("IVF_FLAT"))
			Expect(config.Collection.VectorField.MetricType).To(Equal("L2"))
		})

		It("should support performance tuning", func() {
			configPath := filepath.Join(tempDir, "performance-config.yaml")
			createPerformanceConfig(configPath)

			config, err := cache.LoadMilvusConfig(configPath)
			Expect(err).NotTo(HaveOccurred())

			Expect(config.Performance.ConnectionPool.MaxConnections).To(Equal(20))
			Expect(config.Performance.Batch.InsertBatchSize).To(Equal(500))
			Expect(config.Performance.Batch.Timeout).To(Equal(60))
		})

		It("should support TTL configuration", func() {
			configPath := filepath.Join(tempDir, "ttl-config.yaml")
			createTTLConfig(configPath)

			config, err := cache.LoadMilvusConfig(configPath)
			Expect(err).NotTo(HaveOccurred())

			Expect(config.DataManagement.TTL.Enabled).To(BeTrue())
			Expect(config.DataManagement.TTL.TimestampField).To(Equal("created_at"))
			Expect(config.DataManagement.TTL.CleanupInterval).To(Equal(1800))
		})
	})
})

// Helper functions for creating test configurations

func createValidConfig(path string) {
	config := `connection:
  host: "localhost"
  port: 19530
  timeout: 30
  auth:
    enabled: false
  tls:
    enabled: false
collection:
  name: "test_collection"
  description: "Test collection"
  vector_field:
    name: "embedding"
    dimension: 384
    metric_type: "IP"
  index:
    type: "HNSW"
    params:
      M: 16
      efConstruction: 64
search:
  params:
    ef: 64
  topk: 10
  consistency_level: "Session"
performance:
  connection_pool:
    max_connections: 10
    max_idle_connections: 5
    acquire_timeout: 5
  batch:
    insert_batch_size: 1000
    timeout: 30
data_management:
  ttl:
    enabled: true
    timestamp_field: "timestamp"
    cleanup_interval: 3600
  compaction:
    enabled: true
    interval: 86400
logging:
  level: "info"
  enable_query_log: false
  enable_metrics: true
development:
  drop_collection_on_startup: false
  auto_create_collection: true
  verbose_errors: false`

	err := os.WriteFile(path, []byte(config), 0644)
	Expect(err).NotTo(HaveOccurred())
}

func createInvalidYAMLConfig(path string) {
	config := `connection:
  host: "localhost"
  port: 19530
    invalid yaml structure
collection: [not, a, map]`

	err := os.WriteFile(path, []byte(config), 0644)
	Expect(err).NotTo(HaveOccurred())
}

func createConfigWithInvalidFields(path string) {
	config := `connection:
  host: ""
  port: -1
collection:
  name: "test_collection"
  vector_field:
    name: "embedding"
    dimension: -1`

	err := os.WriteFile(path, []byte(config), 0644)
	Expect(err).NotTo(HaveOccurred())
}

func createMinimalConfig(path string) {
	config := `connection:
  host: "localhost"
  port: 19530
collection:
  name: ""
  vector_field:
    name: ""
    dimension: 0`

	err := os.WriteFile(path, []byte(config), 0644)
	Expect(err).NotTo(HaveOccurred())
}

func createAuthConfigWithoutCredentials(path string) {
	config := `connection:
  host: "localhost"
  port: 19530
  auth:
    enabled: true
    username: ""
    password: ""
collection:
  name: "test_collection"
  vector_field:
    name: "embedding"
    dimension: 384`

	err := os.WriteFile(path, []byte(config), 0644)
	Expect(err).NotTo(HaveOccurred())
}

func createTLSConfigWithoutCert(path string) {
	config := `connection:
  host: "localhost"
  port: 19530
  tls:
    enabled: true
    cert_file: ""
    key_file: ""
collection:
  name: "test_collection"
  vector_field:
    name: "embedding"
    dimension: 384`

	err := os.WriteFile(path, []byte(config), 0644)
	Expect(err).NotTo(HaveOccurred())
}

func createConfigWithInvalidPort(path string) {
	config := `connection:
  host: "localhost"
  port: 99999
collection:
  name: "test_collection"
  vector_field:
    name: "embedding"
    dimension: 384`

	err := os.WriteFile(path, []byte(config), 0644)
	Expect(err).NotTo(HaveOccurred())
}

func createValidAuthConfig(path string) {
	config := `connection:
  host: "localhost"
  port: 19530
  auth:
    enabled: true
    username: "test_user"
    password: "test_password"
collection:
  name: "test_collection"
  vector_field:
    name: "embedding"
    dimension: 384`

	err := os.WriteFile(path, []byte(config), 0644)
	Expect(err).NotTo(HaveOccurred())
}

func createValidTLSConfig(path string) {
	config := `connection:
  host: "localhost"
  port: 19530
  tls:
    enabled: true
    cert_file: "/path/to/cert.pem"
    key_file: "/path/to/key.pem"
    ca_file: "/path/to/ca.pem"
collection:
  name: "test_collection"
  vector_field:
    name: "embedding"
    dimension: 384`

	err := os.WriteFile(path, []byte(config), 0644)
	Expect(err).NotTo(HaveOccurred())
}

func createIVFFlatConfig(path string) {
	config := `connection:
  host: "localhost"
  port: 19530
collection:
  name: "test_collection"
  vector_field:
    name: "embedding"
    dimension: 384
    metric_type: "L2"
  index:
    type: "IVF_FLAT"
    params:
      nlist: 1024
search:
  params:
    nprobe: 10
  topk: 10`

	err := os.WriteFile(path, []byte(config), 0644)
	Expect(err).NotTo(HaveOccurred())
}

func createPerformanceConfig(path string) {
	config := `connection:
  host: "localhost"
  port: 19530
performance:
  connection_pool:
    max_connections: 20
    max_idle_connections: 10
    acquire_timeout: 10
  batch:
    insert_batch_size: 500
    timeout: 60
collection:
  name: "test_collection"
  vector_field:
    name: "embedding"
    dimension: 384`

	err := os.WriteFile(path, []byte(config), 0644)
	Expect(err).NotTo(HaveOccurred())
}

func createTTLConfig(path string) {
	config := `connection:
  host: "localhost"
  port: 19530
data_management:
  ttl:
    enabled: true
    timestamp_field: "created_at"
    cleanup_interval: 1800
collection:
  name: "test_collection"
  vector_field:
    name: "embedding"
    dimension: 384`

	err := os.WriteFile(path, []byte(config), 0644)
	Expect(err).NotTo(HaveOccurred())
}

func TestMilvusConfiguration(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Milvus Configuration Suite")
}
