<script>
export default {
  name: "NewsTab",
  props: {
    news: {
      type: Object,
      required: true,
    },
  },
  data() {
    return {
      defaultImage:
        "https://thumb.ac-illust.com/b1/b170870007dfa419295d949814474ab2_t.jpeg",
    };
  },
};
</script>

<template>
  <div class="card mb-3 shadow-sm">
    <div class="row g-0">
      <div class="col-md-4">
        <img
          :src="news.image || defaultImage"
          class="img-fluid rounded-start"
          alt="News Image"
        />
      </div>
      <div class="col-md-8">
        <div class="card-body">
          <h5 class="card-title">{{ news.title }}</h5>
          <p class="card-text">{{ news.summary }}</p>
          <a
            :href="news.link"
            target="_blank"
            class="btn btn-sm btn-primary me-2"
            >Read More</a
          >
          <span
            class="badge"
            :class="news.pred === 1 ? 'bg-danger' : 'bg-success'"
          >
            {{ news.pred === 1 ? "Fake" : "Real" }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.card-title {
  font-weight: 600;
}
</style>
