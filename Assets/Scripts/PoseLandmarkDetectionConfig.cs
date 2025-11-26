// Copyright (c) 2023 homuler
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

using System.ComponentModel;
using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Vision.PoseLandmarker;
using Mediapipe.Unity;
using Mediapipe.Unity.Sample;
using RunningMode = Mediapipe.Tasks.Vision.Core.RunningMode;

namespace DodgeMan
{
  public enum ModelType : int
  {
    [Description("Pose landmarker (lite)")]
    BlazePoseLite = 0,
    [Description("Pose landmarker (Full)")]
    BlazePoseFull = 1,
    [Description("Pose landmarker (Heavy)")]
    BlazePoseHeavy = 2,
  }

  public class PoseLandmarkDetectionConfig
  {
    public BaseOptions.Delegate Delegate { get; set; } =
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
      BaseOptions.Delegate.CPU;
#else
      BaseOptions.Delegate.GPU;
#endif

    public ImageReadMode ImageReadMode { get; set; } = ImageReadMode.CPUAsync;

    public ModelType Model { get; set; } = ModelType.BlazePoseFull;
    public RunningMode RunningMode { get; set; } = RunningMode.LIVE_STREAM;

    public int NumPoses { get; set; } = 1;
    public float MinPoseDetectionConfidence { get; set; } = 0.5f;
    public float MinPosePresenceConfidence { get; set; } = 0.5f;
    public float MinTrackingConfidence { get; set; } = 0.5f;
    public bool OutputSegmentationMasks { get; set; } = false;
    public string ModelName => Model.GetDescription() ?? Model.ToString();
    public string ModelPath
    {
      get
      {
        switch (Model)
        {
          case ModelType.BlazePoseLite:
            return "pose_landmarker_lite.bytes";
          case ModelType.BlazePoseFull:
            return "pose_landmarker_full.bytes";
          case ModelType.BlazePoseHeavy:
            return "pose_landmarker_heavy.bytes";
          default:
            return null;
        }
      }
    }

    public PoseLandmarkerOptions GetPoseLandmarkerOptions(PoseLandmarkerOptions.ResultCallback resultCallback = null)
    {
      return new PoseLandmarkerOptions(
        new BaseOptions(Delegate, modelAssetPath: ModelPath),
        runningMode: RunningMode,
        numPoses: NumPoses,
        minPoseDetectionConfidence: MinPoseDetectionConfidence,
        minPosePresenceConfidence: MinPosePresenceConfidence,
        minTrackingConfidence: MinTrackingConfidence,
        outputSegmentationMasks: OutputSegmentationMasks,
        resultCallback: resultCallback
      );
    }
  }
}
