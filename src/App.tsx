import { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver, NormalizedLandmark } from '@mediapipe/tasks-vision';
import { ShaderBackground } from './shaderbackground';

interface EventLogItem {
  timestamp: string;
  message: string;
  type: 'info' | 'warning' | 'error' | 'success';
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  
  // High-performance DOM refs to update telemetry without triggering React re-renders
  const yawValRef = useRef<HTMLSpanElement | null>(null);
  const pitchValRef = useRef<HTMLSpanElement | null>(null);
  const leftIrisOffsetRef = useRef<HTMLSpanElement | null>(null);
  const rightIrisOffsetRef = useRef<HTMLSpanElement | null>(null);
  const gazeBarRef = useRef<HTMLDivElement | null>(null);
  const gazeTextRef = useRef<HTMLSpanElement | null>(null);
  
  // MediaPipe and animation refs
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const requestRef = useRef<number | null>(null);
  const gazeAwayStartTimeRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // VeritasEdge States
  const [hasStarted, setHasStarted] = useState(false);

  // React State for macro UI updates (alerts and logs)
  const [modelLoading, setModelLoading] = useState(true);
  const [webcamActive, setWebcamActive] = useState(false);
  const [multipleFaces, setMultipleFaces] = useState(false);
  const [lookingAway, setLookingAway] = useState(false);
  const [eventLogs, setEventLogs] = useState<EventLogItem[]>([]);
  const [statusText, setStatusText] = useState('Initializing...');

  // State refs to track exact status values inside requestAnimationFrame without enclosure issues
  const lookingAwayRef = useRef(false);
  const multipleFacesRef = useRef(false);

  // Telemetry smoothing filters to prevent visual stuttering and erratic alerts
  const smoothedYawRef = useRef<number>(1.0);
  const smoothedPitchRef = useRef<number>(0.6);
  const smoothedLeftIrisRef = useRef<number>(0.0);
  const smoothedRightIrisRef = useRef<number>(0.0);

  // Helper to add log entries
  const addLog = (message: string, type: 'info' | 'warning' | 'error' | 'success') => {
    const timestamp = new Date().toLocaleTimeString();
    setEventLogs((prev) => [{ timestamp, message, type }, ...prev].slice(0, 30));
  };

  // Initialize MediaPipe Face Landmarker
  useEffect(() => {
    let active = true;

    async function initMediaPipe() {
      try {
        setStatusText('Loading browser WebAssembly files...');
        const filesetResolver = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm'
        );

        if (!active) return;
        setStatusText('Loading FaceLandmarker float16 task model...');
        
        const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numFaces: 4,
          outputFaceBlendshapes: true
        });

        if (!active) {
          landmarker.close();
          return;
        }

        faceLandmarkerRef.current = landmarker;
        setModelLoading(false);
        setStatusText('Model ready. Please enable camera.');
        addLog('MediaPipe FaceLandmarker initialized successfully', 'success');
      } catch (err: any) {
        console.error('Failed to load MediaPipe', err);
        setStatusText('Initialization failed: ' + err.message);
        addLog('Model initialization failed: ' + err.message, 'error');
      }
    }

    initMediaPipe();

    return () => {
      active = false;
      if (faceLandmarkerRef.current) {
        faceLandmarkerRef.current.close();
      }
    };
  }, []);

  // Web Camera Stream setup
  const startCamera = async () => {
    if (streamRef.current) {
      stopCamera();
    }

    try {
      addLog('Accessing webcam media stream...', 'info');
      const constraints = {
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setWebcamActive(true);
      setStatusText('Session active. Monitoring...');
      addLog('Webcam connected and playing stream', 'success');
    } catch (err: any) {
      console.error('Camera access error:', err);
      addLog('Camera access denied: ' + err.message, 'error');
      alert('Could not open webcam: Please allow camera access in browser permissions.');
    }
  };

  const stopCamera = () => {
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
      requestRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setWebcamActive(false);
    setMultipleFaces(false);
    setLookingAway(false);
    multipleFacesRef.current = false;
    lookingAwayRef.current = false;
    gazeAwayStartTimeRef.current = null;
    setStatusText('Session paused.');
    addLog('Webcam connection stopped', 'info');
  };

  // Perform processing in the requestAnimationFrame loop
  useEffect(() => {
    if (!webcamActive || modelLoading || !videoRef.current || !canvasRef.current || !hasStarted) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let lastVideoTime = -1;
    const alpha = 0.08; // Telemetry EMA Smoothing Factor (smaller = smoother, larger = responsive)

    const renderLoop = () => {
      if (video.readyState >= 2) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();

        const timestamp = performance.now();
        if (video.currentTime !== lastVideoTime && faceLandmarkerRef.current) {
          lastVideoTime = video.currentTime;
          
          try {
            const results = faceLandmarkerRef.current.detectForVideo(video, timestamp);
            
            // 1. MULTI-FACE DETECTION
            const numFaces = results.faceLandmarks ? results.faceLandmarks.length : 0;
            if (numFaces > 1) {
              if (!multipleFacesRef.current) {
                multipleFacesRef.current = true;
                setMultipleFaces(true);
                addLog(`Multiple faces detected! Count: ${numFaces}`, 'error');
              }
            } else {
              if (multipleFacesRef.current) {
                multipleFacesRef.current = false;
                setMultipleFaces(false);
                addLog('Face count returned to normal.', 'success');
              }
            }

            // 2. GAZE TRACKING & DRAWING
            if (numFaces > 0) {
              const landmarks = results.faceLandmarks[0];

              if (landmarks[33] && landmarks[133] && landmarks[263] && landmarks[362] && landmarks[468] && landmarks[473]) {
                const project = (l: NormalizedLandmark) => ({
                  x: (1 - l.x) * canvas.width,
                  y: l.y * canvas.height
                });

                // Compute Raw Metrics
                const pNose = project(landmarks[1]);
                const pLeftEyeOuter = project(landmarks[33]);
                const pRightEyeOuter = project(landmarks[263]);
                const leftSpan = Math.abs(pNose.x - pLeftEyeOuter.x);
                const rightSpan = Math.abs(pNose.x - pRightEyeOuter.x);
                const yawRatio = rightSpan > 0 ? leftSpan / rightSpan : 1.0;

                const pForehead = project(landmarks[10]);
                const pChin = project(landmarks[152]);
                const upperSpan = Math.abs(pNose.y - pForehead.y);
                const lowerSpan = Math.abs(pNose.y - pChin.y);
                const pitchRatio = lowerSpan > 0 ? upperSpan / lowerSpan : 0.6;

                const pLeftInner = project(landmarks[133]);
                const pLeftOuter = project(landmarks[33]);
                const leftEyeWidth = Math.abs(pLeftInner.x - pLeftOuter.x);
                const leftEyeMid = (pLeftInner.x + pLeftOuter.x) / 2;
                const leftIrisOffset = leftEyeWidth > 0 ? (project(landmarks[468]).x - leftEyeMid) / (leftEyeWidth / 2) : 0;

                const pRightInner = project(landmarks[362]);
                const pRightOuter = project(landmarks[263]);
                const rightEyeWidth = Math.abs(pRightInner.x - pRightOuter.x);
                const rightEyeMid = (pRightInner.x + pRightOuter.x) / 2;
                const rightIrisOffset = rightEyeWidth > 0 ? (project(landmarks[473]).x - rightEyeMid) / (rightEyeWidth / 2) : 0;

                const leftIrisYOffset = leftEyeWidth > 0 ? (project(landmarks[468]).y - (pLeftInner.y + pLeftOuter.y) / 2) / leftEyeWidth : 0;

                // Apply Exponential Moving Average (EMA) to prevent twitching/stuttering
                smoothedYawRef.current = alpha * yawRatio + (1 - alpha) * smoothedYawRef.current;
                smoothedPitchRef.current = alpha * pitchRatio + (1 - alpha) * smoothedPitchRef.current;
                smoothedLeftIrisRef.current = alpha * leftIrisOffset + (1 - alpha) * smoothedLeftIrisRef.current;
                smoothedRightIrisRef.current = alpha * rightIrisOffset + (1 - alpha) * smoothedRightIrisRef.current;

                // Update Telemetry Panel using direct refs (formatted to exactly 3 decimal places)
                if (yawValRef.current) yawValRef.current.innerText = smoothedYawRef.current.toFixed(3);
                if (pitchValRef.current) pitchValRef.current.innerText = smoothedPitchRef.current.toFixed(3);
                if (leftIrisOffsetRef.current) leftIrisOffsetRef.current.innerText = smoothedLeftIrisRef.current.toFixed(3);
                if (rightIrisOffsetRef.current) rightIrisOffsetRef.current.innerText = smoothedRightIrisRef.current.toFixed(3);

                // Determine gaze state based on smoothed metrics
                const isLookAway =
                  smoothedYawRef.current < 0.60 || smoothedYawRef.current > 1.65 ||
                  smoothedPitchRef.current < 0.40 || smoothedPitchRef.current > 1.25 ||
                  Math.abs(smoothedLeftIrisRef.current) > 0.25 ||
                  Math.abs(smoothedRightIrisRef.current) > 0.25 ||
                  leftIrisYOffset > 0.16 || leftIrisYOffset < -0.16;

                // Evaluate Gaze Timer (5-second rule)
                let elapsedSeconds = 0;
                if (isLookAway) {
                  if (gazeAwayStartTimeRef.current === null) {
                    gazeAwayStartTimeRef.current = timestamp;
                    addLog('System alert: User looks away. Timer started.', 'info');
                  }
                  elapsedSeconds = (timestamp - gazeAwayStartTimeRef.current) / 1000;
                  
                  if (elapsedSeconds > 5.0) {
                    if (!lookingAwayRef.current) {
                      lookingAwayRef.current = true;
                      setLookingAway(true);
                      addLog('Session violation: User looking away continuously for > 5 seconds.', 'error');
                    }
                  }
                } else {
                  if (gazeAwayStartTimeRef.current !== null) {
                    gazeAwayStartTimeRef.current = null;
                    addLog('User focus restored to center.', 'success');
                  }
                  if (lookingAwayRef.current) {
                    lookingAwayRef.current = false;
                    setLookingAway(false);
                  }
                }

                // Update timer UI directly
                if (gazeBarRef.current && gazeTextRef.current) {
                  const percent = Math.min((elapsedSeconds / 5) * 100, 100);
                  gazeBarRef.current.style.width = `${percent}%`;
                  gazeTextRef.current.innerText = `${elapsedSeconds.toFixed(1)}s / 5.0s`;

                  if (percent >= 100) {
                    gazeBarRef.current.className = 'progress-fill danger';
                  } else if (percent > 40) {
                    gazeBarRef.current.className = 'progress-fill warning';
                  } else {
                    gazeBarRef.current.className = 'progress-fill safe';
                  }
                }

                // Canvas Rendering of Mesh
                const isViolation = lookingAwayRef.current || multipleFacesRef.current;
                const strokeColor = isViolation ? 'rgba(255, 30, 80, 0.6)' : 'rgba(0, 229, 255, 0.6)';
                
                const drawFeature = (indices: number[], color: string, thickness: number, close: boolean = false) => {
                  ctx.beginPath();
                  const pStart = project(landmarks[indices[0]]);
                  ctx.moveTo(pStart.x, pStart.y);
                  for (let i = 1; i < indices.length; i++) {
                    const p = project(landmarks[indices[i]]);
                    ctx.lineTo(p.x, p.y);
                  }
                  if (close) ctx.closePath();
                  ctx.strokeStyle = color;
                  ctx.lineWidth = thickness;
                  ctx.stroke();
                };

                const leftEyeIndices = [33, 160, 158, 133, 153, 144];
                const rightEyeIndices = [362, 385, 387, 263, 373, 380];
                const outerMouth = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
                const noseBridge = [168, 6, 197, 195, 5];

                drawFeature(leftEyeIndices, 'rgba(0, 255, 150, 0.65)', 2, true);
                drawFeature(rightEyeIndices, 'rgba(0, 255, 150, 0.65)', 2, true);
                drawFeature(outerMouth, strokeColor, 1.5, true);
                drawFeature(noseBridge, strokeColor, 1.5);

                ctx.fillStyle = isLookAway ? '#ff3366' : '#00ffd2';
                const pLeftIris = project(landmarks[468]);
                ctx.beginPath();
                ctx.arc(pLeftIris.x, pLeftIris.y, 4, 0, Math.PI * 2);
                ctx.fill();

                const pRightIris = project(landmarks[473]);
                ctx.beginPath();
                ctx.arc(pRightIris.x, pRightIris.y, 4, 0, Math.PI * 2);
                ctx.fill();

                // Draw Smooth Gaze Direction pointer based on EMA ratios
                const gazeMultiplier = 120;
                const gX = pNose.x + (smoothedLeftIrisRef.current + smoothedRightIrisRef.current) * gazeMultiplier + (smoothedYawRef.current - 1.0) * 110;
                const gY = pNose.y + leftIrisYOffset * 2 * gazeMultiplier + (smoothedPitchRef.current - 0.6) * 110;

                ctx.beginPath();
                ctx.moveTo(pNose.x, pNose.y);
                ctx.lineTo(gX, gY);
                ctx.strokeStyle = isLookAway ? '#ff3366' : '#00ff66';
                ctx.lineWidth = 3.5;
                ctx.stroke();

                ctx.fillStyle = isLookAway ? '#ff3366' : '#00ff66';
                ctx.beginPath();
                ctx.arc(gX, gY, 6, 0, Math.PI * 2);
                ctx.fill();

                const xCoordinates = landmarks.map(l => (1 - l.x) * canvas.width);
                const yCoordinates = landmarks.map(l => l.y * canvas.height);
                const minX = Math.min(...xCoordinates);
                const maxX = Math.max(...xCoordinates);
                const minY = Math.min(...yCoordinates);
                const maxY = Math.max(...yCoordinates);
                
                ctx.strokeStyle = isLookAway ? '#ff3366' : '#00e5ff';
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 4]);
                ctx.strokeRect(minX - 12, minY - 12, (maxX - minX) + 24, (maxY - minY) + 24);
                ctx.setLineDash([]);

                ctx.fillStyle = isLookAway ? '#ff3366' : '#00e5ff';
                ctx.font = '11px system-ui';
                ctx.fillText(`CANDIDATE_01`, minX - 10, minY - 18);
              }
            }
          } catch (err) {
            console.error('Frame parsing error:', err);
          }
        }
      }

      requestRef.current = requestAnimationFrame(renderLoop);
    };

    requestRef.current = requestAnimationFrame(renderLoop);

    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [webcamActive, modelLoading, hasStarted]);

  // Clean-up on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Render full-screen VeritasEdge Landing Page if hasStarted is false
  if (!hasStarted) {
    return (
      <div className="landing-page-wrapper">
        {/* Layer 1: Background Canvas */}
        <div className="landing-bg-layer">
          <ShaderBackground />
        </div>

        {/* Layer 2: Foreground UI Elements */}
        <div className="landing-fg-layer">
          <div className="landing-card">
            <div className="landing-badge">DECENTRALIZED WORKSTATION</div>
            <h1 className="landing-title">VeritasEdge</h1>
            <h2 className="landing-subtitle">Decentralized AI Proctoring Suite</h2>
            <p className="landing-desc">
              Next-generation academic integrity suite offering fully local, real-time gaze monitoring, multi-face presence validation, and anti-jitter telemetry filters. No database records. 100% client privacy.
            </p>
            <button className="landing-btn" onClick={() => setHasStarted(true)}>
              Try Now
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="proctor-app">
      {/* HEADER SECTION */}
      <header className="app-header">
        <div className="brand">
          <div className="pulse-dot"></div>
          <h1>VeritasEdge <span className="badge">AI Proctoring POC</span></h1>
        </div>
        <div className="system-status">
          Status: <strong>{statusText}</strong>
        </div>
      </header>

      {/* ALERT AREA */}
      <div className="alert-container">
        {multipleFaces && (
          <div className="alert-banner critical">
            <span className="alert-icon">⚠️</span>
            <div className="alert-text">
              <h3>Multiple Faces Detected!</h3>
              <p>Academic integrity rule violation. Ensure you are alone in front of the camera.</p>
            </div>
          </div>
        )}
        {lookingAway && (
          <div className="alert-banner warning-banner">
            <span className="alert-icon">👁️</span>
            <div className="alert-text">
              <h3>Attention Alert: Looking Away!</h3>
              <p>Continuous off-screen gaze exceeded 5.0 seconds. Keep your eyes centered on the test.</p>
            </div>
          </div>
        )}
      </div>

      {modelLoading ? (
        <div className="loading-card">
          <div className="spinner"></div>
          <h2>Preparing AI Engine</h2>
          <p>{statusText}</p>
        </div>
      ) : (
        <main className="dashboard-grid">
          {/* CAMERA FEED PANEL */}
          <div className="feed-panel bg-panel">
            <div className="panel-header">
              <h2>Real-Time Video Analytics</h2>
              {!webcamActive ? (
                <button className="btn btn-primary" onClick={startCamera}>
                  Start Monitoring
                </button>
              ) : (
                <button className="btn btn-danger" onClick={stopCamera}>
                  Pause Monitoring
                </button>
              )}
            </div>

            <div className="video-viewport">
              {!webcamActive && (
                <div className="cam-placeholder">
                  <div className="cam-placeholder-icon">📷</div>
                  <p>Webcam stream is currently active inside sandbox. Start monitoring to feed calculations.</p>
                </div>
              )}
              {/* Invisible HTML5 Video serving as raw frame source */}
              <video
                ref={videoRef}
                style={{ display: 'none' }}
                playsInline
                muted
              />
              {/* High-quality analytical canvas overlaying calculations */}
              <canvas
                ref={canvasRef}
                style={{ display: webcamActive ? 'block' : 'none' }}
                className="vision-canvas"
              />
            </div>

            {/* LIVE TIMER SECTION (5s Rule) */}
            <div className="timer-section">
              <div className="timer-info">
                <span>Off-Center Gaze Timer:</span>
                <span ref={gazeTextRef} className="timer-value">0.0s / 5.0s</span>
              </div>
              <div className="timer-progress-bar">
                <div ref={gazeBarRef} className="progress-fill safe" style={{ width: '0%' }}></div>
              </div>
            </div>
          </div>

          <div className="side-dash">
            {/* REAL-TIME TELEMETRY PANEL */}
            <div className="telemetry-panel bg-panel">
              <div className="panel-header">
                <h2>Vision Telemetry</h2>
              </div>
              <div className="telemetry-grid">
                <div className="telemetry-card">
                  <span className="telemetry-label">Eyeball Yaw Ratio</span>
                  <span ref={yawValRef} className="telemetry-value">--</span>
                  <span className="telemetry-desc">Symmetric: ~1.0. Turned face thresholds: &lt; 0.60 | &gt; 1.65</span>
                </div>
                <div className="telemetry-card">
                  <span className="telemetry-label">Eyeball Pitch Ratio</span>
                  <span ref={pitchValRef} className="telemetry-value">--</span>
                  <span className="telemetry-desc">Symmetric: ~0.6. Tilt face thresholds: &lt; 0.40 | &gt; 1.25</span>
                </div>
                <div className="telemetry-card">
                  <span className="telemetry-label">Left Iris Offset</span>
                  <span ref={leftIrisOffsetRef} className="telemetry-value">--</span>
                  <span className="telemetry-desc">Centered: ~0.0. Look-away thresholds: &gt; 0.25</span>
                </div>
                <div className="telemetry-card">
                  <span className="telemetry-label">Right Iris Offset</span>
                  <span ref={rightIrisOffsetRef} className="telemetry-value">--</span>
                  <span className="telemetry-desc">Centered: ~0.0. Look-away thresholds: &gt; 0.25</span>
                </div>
              </div>
            </div>

            {/* EVENT LOG CARD */}
            <div className="logs-panel bg-panel">
              <div className="panel-header">
                <h2>Real-Time Security Audit Logs</h2>
              </div>
              <div className="logs-list">
                {eventLogs.length === 0 ? (
                  <p className="no-logs">Monitoring events will be logged here in real-time...</p>
                ) : (
                  eventLogs.map((log, index) => (
                    <div key={index} className={`log-item log-${log.type}`}>
                      <span className="log-time">[{log.timestamp}]</span>
                      <span className="log-msg">{log.message}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </main>
      )}

      {/* DEVELOPER FOOTER */}
      <footer className="dev-footer">
        Built by Shredev with ❤️
      </footer>
    </div>
  );
}
