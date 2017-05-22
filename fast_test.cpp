#include "precomp.hpp"
#include "fast_score.hpp"


#if defined _MSC_VER
# pragma warning( disable : 4127)
#endif

using namespace std;
using namespace cv;

template<int patternSize>
	void FAST_test(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression, int pixmax, int histSize)
	{
		Mat img = _img.getMat();
		
		const int K = patternSize / 2, N = patternSize + K + 1;
//#if CV_SSE2
//		const int quarterPatternSize = patternSize / 4;
//		(void)quarterPatternSize;
//#endif
		int i, j, k, pixel[25];
		makeOffsets(pixel, (int)img.step, patternSize);

		keypoints.clear();

		threshold = min(max(threshold, 0), 255);

//#if CV_SSE2
//		__m128i delta = _mm_set1_epi8(-128), t = _mm_set1_epi8((char)threshold), K16 = _mm_set1_epi8((char)K);
//		(void)K16;
//		(void)delta;
//		(void)t;
//#endif
		uchar threshold_tab[512];
		for (i = -255; i <= 255; i++)
			threshold_tab[i + 255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

		AutoBuffer<uchar> _buf((img.cols + 16) * 3 * (sizeof(int) + sizeof(uchar)) + 128);
		uchar* buf[3];
		buf[0] = _buf; buf[1] = buf[0] + img.cols; buf[2] = buf[1] + img.cols;
		int* cpbuf[3];
		cpbuf[0] = (int*)alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
		cpbuf[1] = cpbuf[0] + img.cols + 1;
		cpbuf[2] = cpbuf[1] + img.cols + 1;
		memset(buf[0], 0, img.cols * 3);

		for (i = 3; i < img.rows - 2; i++)
		{
			const uchar* ptr = img.ptr<uchar>(i) + 3;
			uchar* curr = buf[(i - 3) % 3];
			int* cornerpos = cpbuf[(i - 3) % 3];
			memset(curr, 0, img.cols);
			int ncorners = 0;

			if (i < img.rows - 3)
			{
				j = 3;
//#if CV_SSE2
//				if (patternSize == 16)
//				{
//					for (; j < img.cols - 16 - 3; j += 16, ptr += 16)
//					{
//						__m128i m0, m1;
//						__m128i v0 = _mm_loadu_si128((const __m128i*)ptr);
//						__m128i v1 = _mm_xor_si128(_mm_subs_epu8(v0, t), delta);
//						v0 = _mm_xor_si128(_mm_adds_epu8(v0, t), delta);
//
//						__m128i x0 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[0])), delta);
//						__m128i x1 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[quarterPatternSize])), delta);
//						__m128i x2 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[2 * quarterPatternSize])), delta);
//						__m128i x3 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[3 * quarterPatternSize])), delta);
//						m0 = _mm_and_si128(_mm_cmpgt_epi8(x0, v0), _mm_cmpgt_epi8(x1, v0));
//						m1 = _mm_and_si128(_mm_cmpgt_epi8(v1, x0), _mm_cmpgt_epi8(v1, x1));
//						m0 = _mm_or_si128(m0, _mm_and_si128(_mm_cmpgt_epi8(x1, v0), _mm_cmpgt_epi8(x2, v0)));
//						m1 = _mm_or_si128(m1, _mm_and_si128(_mm_cmpgt_epi8(v1, x1), _mm_cmpgt_epi8(v1, x2)));
//						m0 = _mm_or_si128(m0, _mm_and_si128(_mm_cmpgt_epi8(x2, v0), _mm_cmpgt_epi8(x3, v0)));
//						m1 = _mm_or_si128(m1, _mm_and_si128(_mm_cmpgt_epi8(v1, x2), _mm_cmpgt_epi8(v1, x3)));
//						m0 = _mm_or_si128(m0, _mm_and_si128(_mm_cmpgt_epi8(x3, v0), _mm_cmpgt_epi8(x0, v0)));
//						m1 = _mm_or_si128(m1, _mm_and_si128(_mm_cmpgt_epi8(v1, x3), _mm_cmpgt_epi8(v1, x0)));
//						m0 = _mm_or_si128(m0, m1);
//						int mask = _mm_movemask_epi8(m0);
//						if (mask == 0)
//							continue;
//						if ((mask & 255) == 0)
//						{
//							j -= 8;
//							ptr -= 8;
//							continue;
//						}
//
//						__m128i c0 = _mm_setzero_si128(), c1 = c0, max0 = c0, max1 = c0;
//						for (k = 0; k < N; k++)
//						{
//							__m128i x = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(ptr + pixel[k])), delta);
//							m0 = _mm_cmpgt_epi8(x, v0);
//							m1 = _mm_cmpgt_epi8(v1, x);
//
//							c0 = _mm_and_si128(_mm_sub_epi8(c0, m0), m0);
//							c1 = _mm_and_si128(_mm_sub_epi8(c1, m1), m1);
//
//							max0 = _mm_max_epu8(max0, c0);
//							max1 = _mm_max_epu8(max1, c1);
//						}
//
//						max0 = _mm_max_epu8(max0, max1);
//						int m = _mm_movemask_epi8(_mm_cmpgt_epi8(max0, K16));
//
//						for (k = 0; m > 0 && k < 16; k++, m >>= 1)
//							if (m & 1)
//							{
//								cornerpos[ncorners++] = j + k;
//								if (nonmax_suppression)
//									curr[j + k] = (uchar)cornerScore<patternSize>(ptr + k, pixel, threshold);
//							}
//					}
//				}
//#endif
				for (; j < img.cols - 3; j++, ptr++)
				{
					int v = ptr[0];
					
				
					const uchar* tab = &threshold_tab[0] - v + 255;
					int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

					if (d == 0)
						continue;

					d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
					d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
					d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

					if (d == 0)
						continue;

					d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
					d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
					d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
					d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];
				if (((pixmax)*(256 / histSize) <= v)&&(v <= (pixmax + 1)*(256 / histSize)))
					{
					//cout << "v:" << v << endl;
					if (d & 1)
					{
						
						
						int vt = v - threshold, count = 0;

						for (k = 0; k < N; k++)
						{
							int x = ptr[pixel[k]];
							if (x < vt)
							{
								if (++count > K)
								{
									cornerpos[ncorners++] = j;
									if (nonmax_suppression)
										curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
									break;
								}
							}
							else
								count = 0;
						}
						
					}

					if (d & 2)
					{
						int vt = v + threshold, count = 0;

						for (k = 0; k < N; k++)
						{
							int x = ptr[pixel[k]];
							if (x > vt)
							{
								if (++count > K)
								{
									cornerpos[ncorners++] = j;
									if (nonmax_suppression)
										curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
									break;
								}
							}
							else
								count = 0;
						}
					}
					}
				}
			}

			cornerpos[-1] = ncorners;

			if (i == 3)
				continue;

			const uchar* prev = buf[(i - 4 + 3) % 3];
			const uchar* pprev = buf[(i - 5 + 3) % 3];
			cornerpos = cpbuf[(i - 4 + 3) % 3];
			ncorners = cornerpos[-1];

			for (k = 0; k < ncorners; k++)
			{
				j = cornerpos[k];
				int score = prev[j];
				if (!nonmax_suppression ||
					(score > prev[j + 1] && score > prev[j - 1] &&
						score > pprev[j - 1] && score > pprev[j] && score > pprev[j + 1] &&
						score > curr[j - 1] && score > curr[j] && score > curr[j + 1]))
				{
					keypoints.push_back(KeyPoint((float)j, (float)(i - 1), 7.f, -1, (float)score));
				}
			}
		}
	}
