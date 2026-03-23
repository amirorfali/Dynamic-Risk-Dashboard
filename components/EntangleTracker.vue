<script setup lang="ts">
import { computed } from 'vue'
import { useNav } from '@slidev/client'

const nav = useNav()
const total = computed(() => nav.total.value || 1)
const current = computed(() => nav.currentPage.value || 1)

const width = 1000
const height = 40
const margin = 30

const nodes = computed(() => {
  const count = Math.max(1, total.value)
  const span = (width - margin * 2) / Math.max(1, count - 1)
  return Array.from({ length: count }, (_, i) => {
    const x = margin + i * span
    const y = height / 2
    return { idx: i + 1, x, y }
  })
})

const paths = computed(() => {
  return nodes.value.slice(0, -1).map((n, i) => {
    const n2 = nodes.value[i + 1]
    const midX = (n.x + n2.x) / 2
    const wobble = (i % 2 === 0 ? -8 : 8)
    const midY = height / 2 + wobble
    return `M ${n.x} ${n.y} Q ${midX} ${midY} ${n2.x} ${n2.y}`
  })
})
</script>

<template>
  <Teleport to="body">
    <div class="entangle-tracker" aria-hidden="true">
      <svg :viewBox="`0 0 ${width} ${height}`" preserveAspectRatio="none">
        <g class="entangle-links">
          <path v-for="(d, i) in paths" :key="i" :d="d" />
        </g>
        <g class="entangle-nodes">
          <image
            v-for="n in nodes"
            :key="`img-${n.idx}`"
            href="/particle.png"
            :x="n.x - 10"
            :y="n.y - 10"
            width="30"
            height="30"
            :opacity="n.idx <= current ? 1 : 0.1"
          />
        </g>
      </svg>
      <div class="entangle-count">{{ current }} / {{ total }}</div>
    </div>
  </Teleport>
</template>
