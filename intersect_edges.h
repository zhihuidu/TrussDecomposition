// Copyright 2022 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef INTERSECT_EDGES_HPP
#define INTERSECT_EDGES_HPP
#include <immintrin.h>

#include <cstdint>
#include <cstdlib>



#include "mmapped_vector.h"   // NOLINT


using edge_t = uint32_t;
using node_t = uint32_t;

using Graph = GraphT<node_t, edge_t>;

template <typename Graph, typename CB>
void IntersectEdgesSmaller(Graph *__restrict__ g, uint64_t start1,
                           uint64_t end1, uint64_t start2, uint64_t end2,
                           const CB &cb) {
  size_t k2 = start2;
  for (size_t k1 = start1; k1 < end1; k1++) {
    if (k2 >= end2) break;
    if (g->adj[k1] < g->adj[k2]) {
      continue;
    }
    if (g->adj[k1] == g->adj[k2]) {
      if (!cb(k1, k2)) return;
      continue;
    }
    size_t offset;
    for (offset = 4; k2 + offset < end2; offset *= 4) {
      if (g->adj[k2 + offset] + 1 > g->adj[k1]) break;
    }
    if (k2 + offset >= end2) {
      offset = end2 - k2;
      size_t lower = k2;
      size_t upper = k2 + offset;
      while (upper > lower + 1) {
        size_t middle = lower + (upper - lower) / 2;
        if (g->adj[middle] >= g->adj[k1]) {
          upper = middle;
        } else {
          lower = middle;
        }
      }
      k2 = upper;
    } else {
      for (; offset > 0; offset >>= 1) {
        if (g->adj[k2 + offset] < g->adj[k1]) {
          k2 += offset;
        }
      }
      k2++;
    }
    if (k2 < end2 && g->adj[k1] == g->adj[k2]) {
      if (!cb(k1, k2)) return;
      continue;
    }
  }
}





// Compute the intersection of two (sorted) adjacency lists, calling `cb` for
// each element in the intersection. If the size of the two adjacency lists is
// significantly different, calls IntersectEdgesSmaller. Otherwise, uses SIMD to
// quickly compute the intersection of the lists.
template <typename Graph, typename CB>
void IntersectEdges(Graph *__restrict__ g, uint64_t start1, uint64_t end1,
                    uint64_t start2, uint64_t end2, const CB &cb) {
  size_t factor = 2;
  if (factor * (end1 - start1) < end2 - start2) {
    return IntersectEdgesSmaller(g, start1, end1, start2, end2, cb);
  }
  if (end1 - start1 > factor * (end2 - start2)) {
    return IntersectEdgesSmaller(
        g, start2, end2, start1, end1,
        [&cb](uint64_t k2, uint64_t k1) { return cb(k1, k2); });
  }
  uint64_t k1 = start1;
  uint64_t k2 = start2;
  // Execute SSE-accelerated version if SSE4.1 is available. If not, run the
  // fall-back code for the last N % 4 elements of the list on the full list.
#ifdef __SSE4_1__
  static const int32_t cyclic_shift1_sse = _MM_SHUFFLE(0, 3, 2, 1);
  static const int32_t cyclic_shift2_sse = _MM_SHUFFLE(1, 0, 3, 2);
  static const int32_t cyclic_shift3_sse = _MM_SHUFFLE(2, 1, 0, 3);

  // trim lengths to be a multiple of 4
  size_t sse_end1 = ((end1 - k1) / 4) * 4 + k1;
  size_t sse_end2 = ((end2 - k2) / 4) * 4 + k2;

  while (k1 < sse_end1 && k2 < sse_end2) {
    __m128i v1_orig = _mm_loadu_si128((__m128i *)&g->adj[k1]);
    __m128i v2_orig = _mm_loadu_si128((__m128i *)&g->adj[k2]);
    __m128i v2 = v2_orig;

    int64_t initial_k = k1;
    int64_t initial_l = k2;
    //[ move pointers
    int32_t a_max = _mm_extract_epi32(v1_orig, 3);
    int32_t b_max = _mm_extract_epi32(v2, 3);
    k1 += (a_max <= b_max) * 4;
    k2 += (a_max >= b_max) * 4;
    //]

    //[ compute mask of common elements
    __m128i cmp_mask1_v1 = _mm_cmpeq_epi32(v1_orig, v2);  // pairwise comparison
    v2 = _mm_shuffle_epi32(v2_orig, cyclic_shift1_sse);   // shuffling
    __m128i cmp_mask2_v1 = _mm_cmpeq_epi32(v1_orig, v2);  // again...
    v2 = _mm_shuffle_epi32(v2_orig, cyclic_shift2_sse);
    __m128i cmp_mask3_v1 = _mm_cmpeq_epi32(v1_orig, v2);  // and again...
    v2 = _mm_shuffle_epi32(v2_orig, cyclic_shift3_sse);
    __m128i cmp_mask4_v1 = _mm_cmpeq_epi32(v1_orig, v2);  // and again.
    __m128i cmp_mask_v1 =
        _mm_or_si128(_mm_or_si128(cmp_mask1_v1, cmp_mask2_v1),
                     _mm_or_si128(cmp_mask3_v1, cmp_mask4_v1));
    int32_t mask_v1 = _mm_movemask_ps((__m128)cmp_mask_v1);

    if (mask_v1) {
      __m128i cmp_mask1_v2 = cmp_mask1_v1;
      __m128i cmp_mask2_v2 = _mm_shuffle_epi32(cmp_mask2_v1, cyclic_shift3_sse);
      __m128i cmp_mask3_v2 = _mm_shuffle_epi32(cmp_mask3_v1, cyclic_shift2_sse);
      __m128i cmp_mask4_v2 = _mm_shuffle_epi32(cmp_mask4_v1, cyclic_shift1_sse);
      __m128i cmp_mask_v2 =
          _mm_or_si128(_mm_or_si128(cmp_mask1_v2, cmp_mask2_v2),
                       _mm_or_si128(cmp_mask3_v2, cmp_mask4_v2));
      int32_t mask_v2 = _mm_movemask_ps((__m128)cmp_mask_v2);

      while (mask_v1) {
        int32_t off1 = __builtin_ctz(mask_v1);
        mask_v1 &= ~(1 << off1);
        int32_t off2 = __builtin_ctz(mask_v2);
        mask_v2 &= ~(1 << off2);

        if (!cb(initial_k + off1, initial_l + off2)) return;
      }
    }
  }
#endif
  if (factor * (end1 - k1) < end2 - k2) {
    return IntersectEdgesSmaller(g, k1, end1, k2, end2, cb);
  }
  if (end1 - k1 > factor * (end2 - k2)) {
    return IntersectEdgesSmaller(
        g, k2, end2, k1, end1,
        [&cb](uint64_t k2, uint64_t k1) { return cb(k1, k2); });
  }
  while (k1 < end1 && k2 < end2) {
    uint64_t a = g->adj[k1];
    uint64_t b = g->adj[k2];
    if (a < b) {
      k1++;
    } else if (a > b) {
      k2++;
    } else {
      if (!cb(k1, k2)) return;
      k1++;
      k2++;
    }
  }
}



template <typename Graph, typename TS, typename CB>
void MinSearchSmaller(Graph *__restrict__ g, TS *__restrict__ s,uint64_t tstart1,
                           uint64_t tend1, uint64_t tstart2, uint64_t tend2,
                           const CB &cb, node_t tv1, node_t tv2) {
  uint64_t start1;
  uint64_t end1;
  uint64_t start2;
  uint64_t end2;
  node_t v1; 
  node_t v2;

  if ((tend1-tstart1) < (tend2-tstart2))  {
       start1=tstart1;
       end1=tend1;
       start2=tstart2;
       end2=tend2;
       v1=tv1;
       v2=tv2;
  } else {
       start2=tstart1;
       end2=tend1;
       start1=tstart2;
       end1=tend2;
       v2=tv1;
       v1=tv2;
  }
  size_t k2 = start2;
#pragma omp parallel for schedule(guided)
  for (size_t k1 = start1; k1 < end1; k1++) {
      node_t v3=g->adj[k1];//get the third vertex, v1<->v2, v1->v3
      if ((v3==v2) || (v3==v1)) {
            continue;
      }

      size_t lower;
      size_t upper;
      size_t middle;

      uint64_t edge2 = s->edge_id[k1];
      uint64_t edge3;
      if ((s->adj_list_end[v3] - g->adj_start[v3]) > (end2-start2)) {
      // search v3 in the adjalency list of v2
          lower=start2;
          upper=end2-1;
          if ((upper<lower) || (v3<g->adj[lower]) || (v3>g->adj[upper]) ) {
               //v1,v2,v3 cannot form a triangle.
               continue;
          }
          if (v3 == g->adj[lower]) {
              if (!cb(k1, lower)) {
                  k1=end1;
              }
              continue;
          } else {
              if (v3==g->adj[upper]) {
                  if (!cb(k1, upper)) {
                      k1=end1;
                  }
                  continue;
              }
          }
          while (upper>lower+1) {
              middle=(lower+upper)/2;
              if (v3 < g->adj[middle]) {
                 upper=middle-1;
                 continue;
              }
              if (v3 > g->adj[middle]) {
                 lower=middle+1;
                 continue;
              }
              if (v3 == g->adj[middle]) {
                  if (!cb(k1, middle)) {
                      k1=end1;
                  } 
                  break;
              }
          }
          if (k1==end1) continue;
          if ((upper<lower) || (v3<g->adj[lower]) || (v3>g->adj[upper]) ) {
               //v1,v2,v3 cannot form a triangle.
               continue;
          }
          if (v3 == g->adj[lower]) {
              if (!cb(k1, lower)) {
                  k1=end1;
              }
              continue;
          } else {
              if (v3==g->adj[upper]) {
                  if (!cb(k1, upper)) {
                      k1=end1;
                  }
                  continue;
              }
          }

      } else {
          // search v2 in adjalency list v3
          lower=g->adj_start[v3];
          upper=s->adj_list_end[v3]-1;
          if ((upper < lower) || (v2 < g->adj[lower]) || ( v2 > g->adj[upper])) {
               continue;
          }
          if (v2 == g->adj[lower]) {
              if (!cb(k1, lower)) {
                  k1=end1;
              }
              continue;
          }
          if (v2 == g->adj[upper]) {
              if (!cb(k1, upper)) {
                  k1=end1;
              }
              continue;
          }
          while (upper>lower+1) {
              middle=(lower+upper)/2;
              if (v2 == g->adj[middle]) {
                  if (!cb(k1, middle)) {
                      k1=end1;
                  }
                  break;
              }
              if (v2 < g->adj[middle]) {
                 upper=middle-1;
                 continue;
              }
              if (v2 > g->adj[middle]) {
                 lower=middle+1;
                 continue;
              }
          }
          if (k1==end1) continue;
          if ((upper < lower) || (v2 < g->adj[lower]) || ( v2 > g->adj[upper])) {
               continue;
          }
          if (v2 == g->adj[lower]) {
              if (!cb(k1, lower)) {
                  k1=end1;
              }
              continue;
          }
          if (v2 == g->adj[upper]) {
              if (!cb(k1, upper)) {
                  k1=end1;
              }
              continue;
          }

      }// end of else
  }//end of for
}//end of MinSearchSmaller




template <typename Graph, typename TS, typename CB>
void SMinSearchSmaller(Graph *__restrict__ g, TS *__restrict__ s,uint64_t start1,
                           uint64_t end1, uint64_t start2, uint64_t end2,
                           const CB &cb, node_t v1, node_t v2) {

  size_t k2 = start2;
#pragma omp parallel for schedule(guided)
  for (size_t k1 = start1; k1 < end1; k1++) {
      node_t v3=g->adj[k1];//get the third vertex, v1<->v2, v1->v3
      if ((v3==v2) || (v3==v1) || v3 < g->adj[start2]) {
            continue;
      }

      size_t lower;
      size_t upper;
      size_t middle;

      uint64_t edge2 = s->edge_id[k1];
      uint64_t edge3;
      if ((s->adj_list_end[v3] - g->adj_start[v3]) > (end2-start2)) {
      // search v3 in the adjalency list of v2
          lower=start2;
          upper=end2-1;
          if ((upper<lower) || (v3<g->adj[lower]) || (v3>g->adj[upper]) ) {
               //v1,v2,v3 cannot form a triangle.
               continue;
          }
          if (v3 == g->adj[lower]) {
              if (!cb(k1, lower)) {
                  k1=end1;
              }
              continue;
          } else {
              if (v3==g->adj[upper]) {
                  if (!cb(k1, upper)) {
                      k1=end1;
                  }
                  continue;
              }
          }
          while (upper>lower+1) {
              middle=(lower+upper)/2;
              if (v3 < g->adj[middle]) {
                 upper=middle-1;
                 continue;
              }
              if (v3 > g->adj[middle]) {
                 lower=middle+1;
                 continue;
              }
              if (v3 == g->adj[middle]) {
                  if (!cb(k1, middle)) {
                      k1=end1;
                  } 
                  break;
              }
          }
          if (k1==end1) continue;
          if ((upper<lower) || (v3<g->adj[lower]) || (v3>g->adj[upper]) ) {
               //v1,v2,v3 cannot form a triangle.
               continue;
          }
          if (v3 == g->adj[lower]) {
              if (!cb(k1, lower)) {
                  k1=end1;
              }
              continue;
          } else {
              if (v3==g->adj[upper]) {
                  if (!cb(k1, upper)) {
                      k1=end1;
                  }
                  continue;
              }
          }

      } else {
          // search v2 in adjalency list v3
          lower=g->adj_start[v3];
          upper=s->adj_list_end[v3]-1;
          if ((upper < lower) || (v2 < g->adj[lower]) || ( v2 > g->adj[upper])) {
               continue;
          }
          if (v2 == g->adj[lower]) {
              if (!cb(k1, lower)) {
                  k1=end1;
              }
              continue;
          }
          if (v2 == g->adj[upper]) {
              if (!cb(k1, upper)) {
                  k1=end1;
              }
              continue;
          }
          while (upper>lower+1) {
              middle=(lower+upper)/2;
              if (v2 == g->adj[middle]) {
                  if (!cb(k1, middle)) {
                      k1=end1;
                  }
                  break;
              }
              if (v2 < g->adj[middle]) {
                 upper=middle-1;
                 continue;
              }
              if (v2 > g->adj[middle]) {
                 lower=middle+1;
                 continue;
              }
          }
          if (k1==end1) continue;
          if ((upper < lower) || (v2 < g->adj[lower]) || ( v2 > g->adj[upper])) {
               continue;
          }
          if (v2 == g->adj[lower]) {
              if (!cb(k1, lower)) {
                  k1=end1;
              }
              continue;
          }
          if (v2 == g->adj[upper]) {
              if (!cb(k1, upper)) {
                  k1=end1;
              }
              continue;
          }

      }// end of else
  }//end of for
}//end of SMinSearchSmaller




// Compute the intersection of two (sorted) adjacency lists based on three adjacency lists, calling `cb` for
// each element in the intersection. 
template <typename Graph,typename TS, typename CB>
void MinSearch(Graph *__restrict__ g, TS *__restrict__ s, uint64_t start1, uint64_t end1,
                    uint64_t start2, uint64_t end2, const CB &cb,node_t v1,node_t v2) {

  return MinSearchSmaller(g,s, start1, end1, start2, end2, cb,v1,v2);

  size_t factor = 2;
  if ( (factor * (end1 - start1) < end2 - start2) ||(end1 - start1 > factor * (end2 - start2))) {
             //fprintf(stderr, "check vertex v1=%4u v2=%4u, start1=%4u, end1=%4u, start2=%4u, end2=%4u\n",v1,v2,start1,end1,start2,end2);

             return MinSearchSmaller(g,s, start1, end1, start2, end2, cb,v1,v2);
  }

  //fprintf(stderr, "two adjacency lists are close to each other");
  uint64_t k1 = start1;
  uint64_t k2 = start2;
  uint64_t    edge2;
  uint64_t    edge3;
  // Execute SSE-accelerated version if SSE4.1 is available. If not, run the
  // fall-back code for the last N % 4 elements of the list on the full list.
#ifdef __SSE4_1__
  static const int32_t cyclic_shift1_sse = _MM_SHUFFLE(0, 3, 2, 1);
  static const int32_t cyclic_shift2_sse = _MM_SHUFFLE(1, 0, 3, 2);
  static const int32_t cyclic_shift3_sse = _MM_SHUFFLE(2, 1, 0, 3);

  // trim lengths to be a multiple of 4
  size_t sse_end1 = ((end1 - k1) / 4) * 4 + k1;
  size_t sse_end2 = ((end2 - k2) / 4) * 4 + k2;

  while (k1 < sse_end1 && k2 < sse_end2) {
    __m128i v1_orig = _mm_loadu_si128((__m128i *)&g->adj[k1]);
    __m128i v2_orig = _mm_loadu_si128((__m128i *)&g->adj[k2]);
    __m128i v2 = v2_orig;

    int64_t initial_k = k1;
    int64_t initial_l = k2;
    //[ move pointers
    int32_t a_max = _mm_extract_epi32(v1_orig, 3);
    int32_t b_max = _mm_extract_epi32(v2, 3);
    k1 += (a_max <= b_max) * 4;
    k2 += (a_max >= b_max) * 4;
    //]

    //[ compute mask of common elements
    __m128i cmp_mask1_v1 = _mm_cmpeq_epi32(v1_orig, v2);  // pairwise comparison
    v2 = _mm_shuffle_epi32(v2_orig, cyclic_shift1_sse);   // shuffling
    __m128i cmp_mask2_v1 = _mm_cmpeq_epi32(v1_orig, v2);  // again...
    v2 = _mm_shuffle_epi32(v2_orig, cyclic_shift2_sse);
    __m128i cmp_mask3_v1 = _mm_cmpeq_epi32(v1_orig, v2);  // and again...
    v2 = _mm_shuffle_epi32(v2_orig, cyclic_shift3_sse);
    __m128i cmp_mask4_v1 = _mm_cmpeq_epi32(v1_orig, v2);  // and again.
    __m128i cmp_mask_v1 =
        _mm_or_si128(_mm_or_si128(cmp_mask1_v1, cmp_mask2_v1),
                     _mm_or_si128(cmp_mask3_v1, cmp_mask4_v1));
    int32_t mask_v1 = _mm_movemask_ps((__m128)cmp_mask_v1);

    if (mask_v1) {
      __m128i cmp_mask1_v2 = cmp_mask1_v1;
      __m128i cmp_mask2_v2 = _mm_shuffle_epi32(cmp_mask2_v1, cyclic_shift3_sse);
      __m128i cmp_mask3_v2 = _mm_shuffle_epi32(cmp_mask3_v1, cyclic_shift2_sse);
      __m128i cmp_mask4_v2 = _mm_shuffle_epi32(cmp_mask4_v1, cyclic_shift1_sse);
      __m128i cmp_mask_v2 =
          _mm_or_si128(_mm_or_si128(cmp_mask1_v2, cmp_mask2_v2),
                       _mm_or_si128(cmp_mask3_v2, cmp_mask4_v2));
      int32_t mask_v2 = _mm_movemask_ps((__m128)cmp_mask_v2);

      while (mask_v1) {
        int32_t off1 = __builtin_ctz(mask_v1);
        mask_v1 &= ~(1 << off1);
        int32_t off2 = __builtin_ctz(mask_v2);
        mask_v2 &= ~(1 << off2);

        //edge2 = s->edge_id[initial_k + off1];
        //edge3 = s->edge_id[initial_l + off2];
        if (!cb(initial_k + off1, initial_l + off2)) return;
      }
    }
  }
#endif
  if ((factor * (end1 - k1) < end2 - k2) ||(end1 - k1 > factor * (end2 - k2))) {
           return MinSearchSmaller(g,s, k1, end1, k2, end2, cb,v1,v2);
  }
  while (k1 < end1 && k2 < end2) {
    uint64_t a = g->adj[k1];
    uint64_t b = g->adj[k2];
    if (a < b) {
      k1++;
    } else if (a > b) {
      k2++;
    } else {
      //edge2 = s->edge_id[k1];
      //edge3 = s->edge_id[k2];
      if (!cb(k1, k2)) return;
      k1++;
      k2++;
    }
  }





}

#endif
