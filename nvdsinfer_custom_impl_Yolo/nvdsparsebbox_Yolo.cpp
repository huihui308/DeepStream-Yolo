/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "nvdsinfer_custom_impl.h"

#include "utils.h"

extern "C" bool
NvDsInferParseV1(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool
NvDsInferParseYoloE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

static NvDsInferParseObjectInfo
convertBBox(const float& bx1, const float& by1, const float& bx2, const float& by2, const uint& netW, const uint& netH)
{
  NvDsInferParseObjectInfo b;

  float x1 = bx1;
  float y1 = by1;
  float x2 = bx2;
  float y2 = by2;

  x1 = clamp(x1, 0, netW);
  y1 = clamp(y1, 0, netH);
  x2 = clamp(x2, 0, netW);
  y2 = clamp(y2, 0, netH);

  b.left = x1;
  b.width = clamp(x2 - x1, 0, netW);
  b.top = y1;
  b.height = clamp(y2 - y1, 0, netH);

  return b;
}

static void
addBBoxProposal(const float bx1, const float by1, const float bx2, const float by2, const uint& netW, const uint& netH,
    const int maxIndex, const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
  NvDsInferParseObjectInfo bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);

  if (bbi.width < 1 || bbi.height < 1) {
      return;
  }

  bbi.detectionConfidence = maxProb;
  bbi.classId = maxIndex;
  binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorYolo(const float* boxes, const float* scores, const float* classes, const uint& outputSize, const uint& netW,
    const uint& netH, const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferParseObjectInfo> binfo;

  for (uint b = 0; b < outputSize; ++b) {
    float maxProb = scores[b];
    int maxIndex = (int) classes[b];

    if (maxProb < preclusterThreshold[maxIndex]) {
      continue;
    }

    float bxc = boxes[b * 4 + 0];
    float byc = boxes[b * 4 + 1];
    float bw = boxes[b * 4 + 2];
    float bh = boxes[b * 4 + 3];

    float bx1 = bxc - bw / 2;
    float by1 = byc - bh / 2;
    float bx2 = bx1 + bw;
    float by2 = by1 + bh;

    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
  }

  return binfo;
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorYoloE(const float* boxes, const float* scores, const float* classes, const uint& outputSize, const uint& netW,
    const uint& netH, const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferParseObjectInfo> binfo;

  for (uint b = 0; b < outputSize; ++b) {
    float maxProb = scores[b];
    int maxIndex = (int) classes[b];

    if (maxProb < preclusterThreshold[maxIndex]) {
      continue;
    }

    float bx1 = boxes[b * 4 + 0];
    float by1 = boxes[b * 4 + 1];
    float bx2 = boxes[b * 4 + 2];
    float by2 = boxes[b * 4 + 3];

    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
  }

  return binfo;
}

static bool
NvDsInferParseCustomYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  std::vector<NvDsInferParseObjectInfo> objects;

  const NvDsInferLayerInfo& boxes = outputLayersInfo[0];
  const NvDsInferLayerInfo& scores = outputLayersInfo[1];
  const NvDsInferLayerInfo& classes = outputLayersInfo[2];

  const uint outputSize = boxes.inferDims.d[0];

  std::vector<NvDsInferParseObjectInfo> outObjs = decodeTensorYolo((const float*) (boxes.buffer),
      (const float*) (scores.buffer), (const float*) (classes.buffer), outputSize, networkInfo.width, networkInfo.height,
      detectionParams.perClassPreclusterThreshold);

  objects.insert(objects.end(), outObjs.begin(), outObjs.end());

  objectList = objects;

  return true;
}

static bool
NvDsInferParseCustomYoloV10(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  std::vector<NvDsInferParseObjectInfo> objects;

  const NvDsInferLayerInfo& boxes = outputLayersInfo[0];
  //const NvDsInferLayerInfo& scores = outputLayersInfo[1];
  //const NvDsInferLayerInfo& classes = outputLayersInfo[2];
  const uint outputSize = boxes.inferDims.d[0];
  float *result = (float*)(boxes.buffer);

  // Process each detection result from the output
  for (uint i = 0; i < outputSize; i++) {
    // Compute the starting index for the current detection result in the 'result' array
    int s = 6 * i;
    int maxIndex = (int)result[s + 5];
    float maxProb = (float)result[s + 4];

    if (maxProb < detectionParams.perClassPreclusterThreshold[maxIndex]) {
      continue;
    }
#if 0
    if ((8 == maxIndex) || (9 == maxIndex) || (10 == maxIndex)) {
      continue;
    }
#endif
    // Extract the coordinates and dimensions of the bounding box (normalized values)
    float cx = result[s + 0];  // Top-left x-coordinate
    float cy = result[s + 1];  // Top-left y-coordinate
    float dx = result[s + 2];  // Bottom-right x-coordinate
    float dy = result[s + 3];  // Bottom-right y-coordinate

    addBBoxProposal(cx, cy, dx, dy, networkInfo.width, networkInfo.height, maxIndex, maxProb, objects);
  }
  objectList = objects;

  return true;
}

static bool
NvDsInferParseCustomYoloE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  std::vector<NvDsInferParseObjectInfo> objects;

  const NvDsInferLayerInfo& boxes = outputLayersInfo[0];
  const NvDsInferLayerInfo& scores = outputLayersInfo[1];
  const NvDsInferLayerInfo& classes = outputLayersInfo[2];

  const uint outputSize = boxes.inferDims.d[0];

  std::vector<NvDsInferParseObjectInfo> outObjs = decodeTensorYoloE((const float*) (boxes.buffer),
      (const float*) (scores.buffer), (const float*) (classes.buffer), outputSize, networkInfo.width, networkInfo.height,
      detectionParams.perClassPreclusterThreshold);

  objects.insert(objects.end(), outObjs.begin(), outObjs.end());

  objectList = objects;

  return true;
}

extern "C" bool
NvDsInferParseV1(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYolo(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParseYoloE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYoloE(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParseV2(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYoloV10(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseV1);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloE);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseV2);
