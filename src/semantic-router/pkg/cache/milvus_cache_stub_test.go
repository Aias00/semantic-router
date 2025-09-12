//go:build !milvus

package cache_test

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Milvus Cache Stub", func() {
	Describe("Stub Implementation", func() {
		It("should handle disabled cache gracefully", func() {
			stubCache, err := cache.NewMilvusCache(cache.MilvusCacheOptions{
				Enabled: false,
			})
			Expect(err).NotTo(HaveOccurred())
			defer stubCache.Close()

			Expect(stubCache.IsEnabled()).To(BeFalse())

			// All operations should return error messages about milvus not being available
			_, _, err = stubCache.FindSimilar("test-model", "test query")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not available"))

			err = stubCache.AddEntry("test-model", "test query", []byte("request"), []byte("response"))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not available"))

			_, err = stubCache.AddPendingRequest("test-model", "test query", []byte("request"))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not available"))

			err = stubCache.UpdateWithResponse("test query", []byte("response"))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("not available"))
		})

		It("should return appropriate stats for disabled cache", func() {
			stubCache, err := cache.NewMilvusCache(cache.MilvusCacheOptions{
				Enabled: false,
			})
			Expect(err).NotTo(HaveOccurred())
			defer stubCache.Close()

			stats := stubCache.GetStats()
			Expect(stats.TotalEntries).To(Equal(0))
			Expect(stats.HitCount).To(Equal(int64(0)))
			Expect(stats.MissCount).To(Equal(int64(0)))
			Expect(stats.HitRatio).To(Equal(0.0))
		})

		It("should close gracefully", func() {
			stubCache, err := cache.NewMilvusCache(cache.MilvusCacheOptions{
				Enabled: false,
			})
			Expect(err).NotTo(HaveOccurred())

			err = stubCache.Close()
			Expect(err).NotTo(HaveOccurred())
		})
	})
})

func TestMilvusCacheStub(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Milvus Cache Stub Suite")
}
