// Copyright (c) 2023 homuler
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Vision.HandLandmarker;
using Mediapipe.Unity;
using RunningMode = Mediapipe.Tasks.Vision.Core.RunningMode;

namespace MediaPipeUnity.Scenes
{
  public class HandLandmarkDetectionConfig
  {
    public BaseOptions.Delegate Delegate { get; set; } =
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
      BaseOptions.Delegate.CPU;
#else
      BaseOptions.Delegate.GPU;
#endif
   
    public ImageReadMode ImageReadMode { get; set; } = ImageReadMode.CPUAsync;
    public RunningMode RunningMode { get; set; } = RunningMode.LIVE_STREAM;

    public int NumHands { get; set; } = 1;
    public float MinHandDetectionConfidence { get; set; } = 0.5f;
    public float MinHandPresenceConfidence { get; set; } = 0.5f;
    public float MinTrackingConfidence { get; set; } = 0.5f;
    public string ModelPath => "hand_landmarker.bytes";

    public HandLandmarkerOptions GetHandLandmarkerOptions(HandLandmarkerOptions.ResultCallback resultCallback = null)
    {
      return new HandLandmarkerOptions(
        new BaseOptions(Delegate, modelAssetPath: ModelPath),
        runningMode: RunningMode,
        numHands: NumHands,
        minHandDetectionConfidence: MinHandDetectionConfidence,
        minHandPresenceConfidence: MinHandPresenceConfidence,
        minTrackingConfidence: MinTrackingConfidence,
        resultCallback: resultCallback
      );
    }
  }
}
