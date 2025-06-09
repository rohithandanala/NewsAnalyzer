<script setup>
import HeadingComponent from "./components/HeadingComponent.vue";
import NewsTab from "./components/NewsTab.vue";

import { ref, onMounted, computed } from "vue";

const news = ref([]);
const loading = ref(true);

const formattedJson = computed(() => {
  return JSON.stringify(news.value, null, 2);
});

onMounted(async () => {
  try {
    const response = await fetch(
      "http://ec2-34-203-226-187.compute-1.amazonaws.com/get_data"
    );
    if (!response.ok) throw new Error("Network response was not ok");
    const data = await response.json();
    news.value = data;
  } catch (error) {
    console.error("Failed to fetch data:", error);
  } finally {
    loading.value = false;
  }
});
</script>

<template>
  <HeadingComponent />
  <div class="container mt-4">
    <div v-if="loading" class="text-center">Loading...</div>
    <div v-else>
      <NewsTab v-for="(item, index) in news" :key="index" :news="item" />
    </div>
  </div>
</template>

<style scoped></style>
