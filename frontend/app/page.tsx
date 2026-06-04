/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import React, { useState, useRef, useEffect } from 'react';
import LiveBackground from './components/LiveBackground';

// The FastAPI backend URL pointing to your Hugging Face Space
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "https://Kritzzz11-bio-acoustica.hf.space";

interface Detection {
  label: string;
  score: number;
  class_name: string;
  sci_name: string;
}

interface RouterWeights {
  Aves: number;
  Insecta: number;
  Amphibia: number;
  Mammalia: number;
  Reptilia: number;
}

interface PredictionResponse {
  detections: Detection[];
  router_weights: RouterWeights;
  spectrogram: number[][];
  n_windows: number;
  duration_sec: number;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('');
  const [results, setResults] = useState<PredictionResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'detections' | 'taxamoe' | 'spectrogram' | 'pipeline'>('detections');
  const [error, setError] = useState<string | null>(null);

  // Audio playback tracking
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const requestRef = useRef<number | null>(null);

  // Drag & Drop states
  const [dragActive, setDragActive] = useState(false);

  // Scroll listener for floating pill navbar
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Cleanup audio url on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [audioUrl]);

  // Handle Drag & Drop
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
      if (selectedFile.type.startsWith('audio/') || selectedFile.name.endsWith('.ogg')) {
        setFile(selectedFile);
        setAudioUrl(URL.createObjectURL(selectedFile));
        setResults(null);
        setError(null);
      } else {
        setError("Please upload a valid audio file (.wav, .mp3, .ogg)");
      }
    }
  };

  // Handle File Input
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setAudioUrl(URL.createObjectURL(selectedFile));
      setResults(null);
      setError(null);
    }
  };

  // Run the stages simulation for realistic premium loading feedback
  const simulateLoadingStages = (stages: string[], callback: () => Promise<void>) => {
    let currentStageIndex = 0;
    setLoadingStage(stages[0]);

    const interval = setInterval(() => {
      if (currentStageIndex < stages.length - 1) {
        currentStageIndex++;
        setLoadingStage(stages[currentStageIndex]);
      }
    }, 2500);

    callback().finally(() => {
      clearInterval(interval);
    });
  };

  // Handle Submission
  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResults(null);
    setIsPlaying(false);

    const stages = [
      "Uploading soundscape to Wilderness Space...",
      "Slicing audio into 5-second temporal windows...",
      "Extracting 1536-dim Perch embeddings via ONNX Runtime...",
      "Routing features through TaxaMoE Adapter...",
      "Running specialized prototypical experts...",
      "Ensembling with ProtoSSM and Sound Event Detection (SED)...",
      "Applying logit-space Bayesian prior & smoothing..."
    ];

    const performPrediction = async () => {
      try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(`${BACKEND_URL}/predict`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          throw new Error(`Server returned ${res.status}: ${res.statusText}`);
        }

        const data: PredictionResponse = await res.json();
        setResults(data);
        setActiveTab('detections');
      } catch (err: any) {
        console.error(err);
        setError(err.message || "An unexpected error occurred while communicating with the model server.");
      } finally {
        setLoading(false);
      }
    };

    simulateLoadingStages(stages, performPrediction);
  };

  // Render Spectrogram on Canvas
  useEffect(() => {
    if (!results || !canvasRef.current || !results.spectrogram) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const spec = results.spectrogram; // Matrix of size [frequencies, time_frames]
    const rows = spec.length;         // Frequencies
    const cols = spec[0].length;      // Time frames

    // Set canvas internal dimensions to match display size
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    const cellWidth = canvas.width / cols;
    const cellHeight = canvas.height / rows;

    // Draw spectrogram matrix
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = spec[r][c]; // Values are in dB (typically negative or normalized)

        // Normalize dB value to 0.0 - 1.0 range
        // Spec is normally -80 to 0
        const normalized = Math.min(Math.max((val + 80) / 80, 0), 1);

        // Color palette mapping: deep dark green to bright gold
        // RGB start (jungle green): 2, 24, 8
        // RGB end (bright gold): 212, 175, 55
        const rColor = Math.round(2 + normalized * (212 - 2));
        const gColor = Math.round(24 + normalized * (175 - 24));
        const bColor = Math.round(8 + normalized * (55 - 8));

        ctx.fillStyle = `rgb(${rColor}, ${gColor}, ${bColor})`;

        // Draw rectangle (flip y-axis since 0 Hz is at the bottom)
        ctx.fillRect(
          c * cellWidth,
          canvas.height - (r + 1) * cellHeight,
          cellWidth + 0.5,
          cellHeight + 0.5
        );
      }
    }
  }, [results, activeTab]);

  // Audio Playback Hook
  const handlePlayPause = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  // Update canvas cursor on playing
  const updatePlaybackPosition = () => {
    if (!audioRef.current || !canvasRef.current || !results) return;

    const audio = audioRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    setPlaybackTime(audio.currentTime);

    // Redraw a tracking line
    const progress = audio.currentTime / audio.duration;
    const cursorX = progress * canvas.width;

    // Clear and redraw is handled by saving the base state, but here we can just overlay
    // For simplicity, we draw the cursor as a thin glowing vertical line
    // To do this cleanly without wiping, we can redraw the spectrogram or just overlay.
    // Overriding is fine, but to keep performance, overlay a gold bar.

    // Draw the tracking bar
    ctx.shadowBlur = 10;
    ctx.shadowColor = "#d4af37";
    ctx.strokeStyle = "#d4af37";
    ctx.lineWidth = 2;

    // We want to redraw to avoid drawing lines everywhere.
    // A quick way is to trigger canvas redraw first.
    const spec = results.spectrogram;
    const rows = spec.length;
    const cols = spec[0].length;
    const cellWidth = canvas.width / cols;
    const cellHeight = canvas.height / rows;

    ctx.shadowBlur = 0; // Turn off shadow for spec cells
    for (let r = 0; r < rows; r++) {
      const idx = Math.floor(progress * cols);
      // We only redraw the neighborhood around the cursor to avoid full redrawing
      const start = Math.max(0, idx - 10);
      const end = Math.min(cols, idx + 10);
      for (let c = start; c < end; c++) {
        const val = spec[r][c];
        const normalized = Math.min(Math.max((val + 80) / 80, 0), 1);
        const rColor = Math.round(2 + normalized * (212 - 2));
        const gColor = Math.round(24 + normalized * (175 - 24));
        const bColor = Math.round(8 + normalized * (55 - 8));
        ctx.fillStyle = `rgb(${rColor}, ${gColor}, ${bColor})`;
        ctx.fillRect(
          c * cellWidth,
          canvas.height - (r + 1) * cellHeight,
          cellWidth + 0.5,
          cellHeight + 0.5
        );
      }
    }

    // Now draw cursor
    ctx.shadowBlur = 10;
    ctx.shadowColor = "#d4af37";
    ctx.beginPath();
    ctx.moveTo(cursorX, 0);
    ctx.lineTo(cursorX, canvas.height);
    ctx.stroke();
    ctx.shadowBlur = 0; // reset

    if (isPlaying) {
      requestRef.current = requestAnimationFrame(updatePlaybackPosition);
    }
  };

  useEffect(() => {
    if (isPlaying) {
      requestRef.current = requestAnimationFrame(updatePlaybackPosition);
    } else {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isPlaying]);

  const handleAudioEnded = () => {
    setIsPlaying(false);
    setPlaybackTime(0);
  };

  // Scroll to workspace helper
  const scrollToWorkspace = () => {
    const el = document.getElementById('workspace');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div style={{ backgroundColor: '#030a06', color: '#e2f0e8', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* Immersive Cinematic Hero Section */}
      <header style={{
        position: 'relative',
        height: '100vh',
        width: '100vw',
        backgroundImage: `radial-gradient(circle at center, rgba(3, 10, 6, 0.4) 0%, #030a06 90%), url('/dense_jungle_hero.png')`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        padding: '0 24px',
        textAlign: 'center',
        overflow: 'hidden'
      }}>

        {/* Live animated jungle components */}
        <div className="live-bg-container">
          <div className="mist-layer mist-1"></div>
          <div className="mist-layer mist-2"></div>
          <div className="sunbeams"></div>
          <LiveBackground />
        </div>

        {/* Glowing glass navbar */}
        <nav className="glass" style={{
          position: 'fixed',
          top: isScrolled ? '16px' : '24px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: 'calc(100% - 48px)',
          maxWidth: isScrolled ? '700px' : '1200px',
          padding: isScrolled ? '12px 24px' : '16px 32px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          zIndex: 100,
          background: isScrolled ? 'rgba(8, 26, 17, 0.75)' : 'transparent',
          border: isScrolled ? '1px solid rgba(18, 184, 99, 0.25)' : '1px solid transparent',
          boxShadow: isScrolled ? '0 12px 32px rgba(0, 0, 0, 0.5), 0 0 15px rgba(18, 184, 99, 0.1)' : 'none',
          borderRadius: isScrolled ? '50px' : '16px',
          backdropFilter: isScrolled ? 'blur(12px)' : 'none',
          WebkitBackdropFilter: isScrolled ? 'blur(12px)' : 'none',
          transition: 'all 0.4s cubic-bezier(0.16, 1, 0.3, 1)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: isScrolled ? '1.2rem' : '1.5rem', color: '#12b863', fontWeight: 'bold', textShadow: '0 0 10px rgba(18,184,99,0.4)', transition: '0.4s' }}>🍃</span>
            <span style={{
              fontFamily: 'var(--font-serif)',
              fontSize: isScrolled ? '1.1rem' : '1.4rem',
              fontWeight: 600,
              letterSpacing: '2px',
              color: '#fff',
              transition: '0.4s'
            }}>WILD ECHO</span>
          </div>

          <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
            <a href="#pantanal" style={{ fontSize: '0.85rem', color: '#8ea79b', transition: '0.3s' }} className="hover-glow">Pantanal</a>
            <a href="#about" style={{ fontSize: '0.85rem', color: '#8ea79b', transition: '0.3s' }} className="hover-glow">Architecture</a>
            <a href="#kaggle" style={{ fontSize: '0.85rem', color: '#8ea79b', transition: '0.3s' }} className="hover-glow">Kaggle</a>
            <a href="#workspace" style={{
              fontSize: '0.85rem',
              color: '#12b863',
              fontWeight: 600,
              textShadow: '0 0 8px rgba(18,184,99,0.3)',
              padding: isScrolled ? '6px 16px' : '8px 20px',
              border: '1px solid rgba(18,184,99,0.2)',
              borderRadius: '20px',
              background: 'rgba(18, 184, 99, 0.05)',
              transition: 'all 0.3s'
            }} className="hover-scale" onClick={(e) => { e.preventDefault(); scrollToWorkspace(); }}>Analyze</a>
          </div>
        </nav>

        {/* Hero Title */}
        <div style={{ maxWidth: '900px', marginTop: '60px', zIndex: 2 }}>
          <h1 className="glow-text" style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 'calc(2.2rem + 2.5vw)',
            lineHeight: 1.1,
            fontWeight: 800,
            color: '#ffffff',
            letterSpacing: '1px',
            marginBottom: '20px'
          }}>
            LISTEN TO THE WHISPERS OF <br />
            <span style={{
              color: 'transparent',
              backgroundImage: 'linear-gradient(to right, #12b863, #d4af37)',
              WebkitBackgroundClip: 'text',
              backgroundClip: 'text'
            }}>THE WETLANDS</span>
          </h1>

          <p style={{
            fontSize: 'calc(1rem + 0.3vw)',
            color: '#c2dfcf',
            maxWidth: '680px',
            margin: '0 auto 40px auto',
            lineHeight: 1.6,
            fontWeight: 300
          }}>
            An advanced bioacoustic intelligence platform mapping the Pantanal soundscapes. Powered by TaxaMoE to detect birds, amphibians, insects, mammals, and reptiles with extreme calibration.
          </p>

          <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
            <button
              onClick={scrollToWorkspace}
              style={{
                background: 'linear-gradient(135deg, #12b863 0%, #0d8c4a 100%)',
                color: '#fff',
                padding: '16px 36px',
                borderRadius: '50px',
                fontSize: '1rem',
                fontWeight: 600,
                border: 'none',
                cursor: 'pointer',
                boxShadow: '0 8px 24px rgba(18, 184, 99, 0.35)',
                transition: '0.3s'
              }}
              className="hover-scale"
            >
              Launch Analyser
            </button>
            <a
              href="#about"
              style={{
                background: 'rgba(255, 255, 255, 0.05)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                color: '#fff',
                padding: '16px 36px',
                borderRadius: '50px',
                fontSize: '1rem',
                fontWeight: 500,
                transition: '0.3s'
              }}
              className="hover-bg-glass"
            >
              Explore Tech Stack
            </a>
          </div>
        </div>

        {/* Shadow overlays */}
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          width: '100%',
          height: '150px',
          backgroundImage: 'linear-gradient(to top, #030a06, transparent)',
          zIndex: 1
        }} />
      </header>

      {/* The Pantanal Wetlands & Project Story Section */}
      <section id="pantanal" style={{
        padding: '100px 24px',
        maxWidth: '1200px',
        margin: '0 auto',
        width: '100%',
        borderBottom: '1px solid rgba(18, 184, 99, 0.15)',
        zIndex: 2,
        position: 'relative'
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: '48px', alignItems: 'center' }}>

          {/* Left Column: Geographical Info & Story */}
          <div>
            <span style={{ color: '#d4af37', textTransform: 'uppercase', fontSize: '0.85rem', fontWeight: 600, letterSpacing: '3px', display: 'block', marginBottom: '8px' }}>THE EPICENTER OF BIODIVERSITY</span>
            <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: 'calc(1.8rem + 1.2vw)', color: '#fff', lineHeight: 1.2, marginBottom: '24px' }}>
              THE PANTANAL <br />
              <span style={{ color: '#12b863' }}>WETLANDS</span>
            </h2>

            <p style={{ color: '#8ea79b', marginBottom: '16px', lineHeight: 1.7, fontSize: '1rem' }}>
              Located in the heart of South America, the <strong>Pantanal</strong> is the world&apos;s largest tropical wetland ecosystem. Spanning over 150,000 square kilometers across Brazil, Bolivia, and Paraguay, it is a mosaic of flooded grasslands, savannas, and tropical forests.
            </p>
            <p style={{ color: '#8ea79b', marginBottom: '24px', lineHeight: 1.7, fontSize: '1rem' }}>
              Unlike dense rainforests, the open wetlands make it a haven for observing wildlife. However, the seasonal flooding cycles make physical travel impossible, rendering <strong>Passive Acoustic Monitoring (PAM)</strong> the only viable method for continuous ecological surveying.
            </p>

            <h3 style={{ fontFamily: 'var(--font-serif)', color: '#fff', fontSize: '1.3rem', marginBottom: '12px' }}>Species of the Soundscape</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div className="glass" style={{ padding: '16px', background: 'rgba(8, 26, 17, 0.25)' }}>
                <strong style={{ color: '#d4af37', fontSize: '0.95rem' }}>Hyacinth Macaw</strong>
                <p style={{ fontSize: '0.8rem', color: '#8ea79b', marginTop: '4px' }}>Large cobalt-blue parrots whose raucous calls dominate the canopy.</p>
              </div>
              <div className="glass" style={{ padding: '16px', background: 'rgba(8, 26, 17, 0.25)' }}>
                <strong style={{ color: '#d4af37', fontSize: '0.95rem' }}>Jaguar</strong>
                <p style={{ fontSize: '0.8rem', color: '#8ea79b', marginTop: '4px' }}>Apex felines whose low-frequency grunts are recorded at night.</p>
              </div>
              <div className="glass" style={{ padding: '16px', background: 'rgba(8, 26, 17, 0.25)' }}>
                <strong style={{ color: '#d4af37', fontSize: '0.95rem' }}>Yacare Caiman</strong>
                <p style={{ fontSize: '0.8rem', color: '#8ea79b', marginTop: '4px' }}>Aquatic reptiles communicating with deep, guttural barks.</p>
              </div>
              <div className="glass" style={{ padding: '16px', background: 'rgba(8, 26, 17, 0.25)' }}>
                <strong style={{ color: '#d4af37', fontSize: '0.95rem' }}>Horned Frog</strong>
                <p style={{ fontSize: '0.8rem', color: '#8ea79b', marginTop: '4px' }}>Amphibians initiating high-pitched trills during heavy rains.</p>
              </div>
            </div>
          </div>

          {/* Right Column: Geographic Map Pinpoint */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div className="glass" style={{
              padding: '16px',
              border: '1px solid rgba(18, 184, 99, 0.25)',
              boxShadow: '0 12px 40px rgba(0, 0, 0, 0.6)',
              position: 'relative',
              overflow: 'hidden',
              borderRadius: '20px',
              width: '100%'
            }}>
              {/* Technical overlay HUD elements */}
              <div style={{
                position: 'absolute',
                top: '12px',
                left: '16px',
                display: 'flex',
                gap: '8px',
                fontSize: '0.7rem',
                color: '#12b863',
                fontFamily: 'monospace',
                zIndex: 2,
                textShadow: '0 0 5px rgba(18, 184, 99, 0.5)'
              }}>
                <span>SYS.LOC // SOUTH_AMERICA</span>
                <span style={{ color: '#d4af37' }}>● ACTIVE</span>
              </div>

              <div style={{
                position: 'absolute',
                bottom: '12px',
                right: '16px',
                fontSize: '0.7rem',
                color: '#8ea79b',
                fontFamily: 'monospace',
                zIndex: 2
              }}>
                COORDS: 18°00&apos;S, 56°00&apos;W
              </div>

              {/* Map image with slight zoom and overlay glow */}
              <div style={{
                borderRadius: '12px',
                overflow: 'hidden',
                position: 'relative',
                height: '350px',
                width: '100%',
                backgroundColor: '#020804'
              }}>
                <img
                  src="/pantanal_map.png"
                  alt="Pantanal Geographic Map Pinpoint"
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    opacity: 0.85
                  }}
                />

                {/* Target overlay glow */}
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  boxShadow: 'inset 0 0 30px rgba(18, 184, 99, 0.15)',
                  pointerEvents: 'none'
                }} />
              </div>
            </div>

            {/* Story text beneath map */}
            <div style={{ marginTop: '24px', textAlign: 'center', maxWidth: '480px' }}>
              <span style={{ color: '#d4af37', fontWeight: 600, fontSize: '0.9rem', display: 'block', marginBottom: '6px' }}>
                Bridging the Taxonomical Gap
              </span>
              <p style={{ color: '#8ea79b', fontSize: '0.85rem', lineHeight: 1.6 }}>
                Wildlife audio classification is heavily skewed towards avian species. WildEcho blends adapter embeddings and custom prototypical experts to ensure all members of the soundscape are recognized.
              </p>
            </div>
          </div>

        </div>
      </section>

      {/* Main Content Workspace */}
      <main id="workspace" style={{ flex: 1, padding: '80px 24px', maxWidth: '1200px', margin: '0 auto', width: '100%' }}>

        {/* Workspace Title */}
        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '2.5rem', color: '#fff', marginBottom: '12px' }}>Acoustic Monitoring Laboratory</h2>
          <p style={{ color: '#8ea79b', maxWidth: '600px', margin: '0 auto' }}>Upload an audio file containing wildlife calls. Our ensemble pipelines will classify species, isolate signals, and route them through our TaxaMoE system.</p>
        </div>

        {/* Upload Card / Interface */}
        <div className="glass" style={{ padding: '32px', marginBottom: '48px', position: 'relative' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', alignItems: 'center' }}>

            {/* Left side: Upload area */}
            <div>
              <div
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                style={{
                  border: `2px dashed ${dragActive ? 'var(--primary)' : 'rgba(18, 184, 99, 0.2)'}`,
                  borderRadius: '12px',
                  padding: '40px 24px',
                  textAlign: 'center',
                  backgroundColor: dragActive ? 'rgba(18, 184, 99, 0.05)' : 'rgba(3, 10, 6, 0.25)',
                  cursor: 'pointer',
                  transition: 'var(--transition-smooth)'
                }}
                onClick={() => document.getElementById('audio-upload')?.click()}
              >
                <input
                  aria-label="input"
                  type="file"
                  id="audio-upload"
                  style={{ display: 'none' }}
                  accept="audio/*,.ogg"
                  onChange={handleFileChange}
                />
                <span style={{ fontSize: '3rem', display: 'block', marginBottom: '16px' }}>🎵</span>
                <h3 style={{ fontSize: '1.2rem', color: '#fff', marginBottom: '8px' }}>Drag & drop soundscape</h3>
                <p style={{ fontSize: '0.85rem', color: '#8ea79b' }}>Supports WAV, MP3, or OGG up to 60 seconds</p>

                {file && (
                  <div style={{
                    marginTop: '20px',
                    padding: '8px 16px',
                    background: 'rgba(18, 184, 99, 0.1)',
                    border: '1px solid rgba(18, 184, 99, 0.2)',
                    borderRadius: '8px',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    <span style={{ fontSize: '0.85rem', color: '#12b863', fontWeight: 600 }}>✓ {file.name}</span>
                  </div>
                )}
              </div>

              {error && (
                <div style={{ marginTop: '16px', color: '#ff6b6b', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>⚠️</span> {error}
                </div>
              )}
            </div>

            {/* Right side: Controls & Info */}
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'space-between' }}>
              <div>
                <h3 style={{ fontSize: '1.3rem', color: '#fff', marginBottom: '12px', fontFamily: 'var(--font-serif)' }}>Pipeline Setup</h3>
                <ul style={{ listStyle: 'none', color: '#8ea79b', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '24px' }}>
                  <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: '#12b863' }}>✔</span> Model Hub: <code style={{ color: '#d4af37', background: 'rgba(212, 175, 55, 0.08)', padding: '2px 6px', borderRadius: '4px' }}>Kritzzz11/bio-acosutica</code>
                  </li>
                  <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: '#12b863' }}>✔</span> Front-ends: Perch v2, ProtoSSM v4, SED (fold0, fold2), TaxaMoE
                  </li>
                  <li style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: '#12b863' }}>✔</span> Target Site: Pantanal Wetlands, Brazil
                  </li>
                </ul>
              </div>

              <div style={{ display: 'flex', gap: '16px' }}>
                <button
                  onClick={handleSubmit}
                  disabled={!file || loading}
                  style={{
                    flex: 1,
                    background: file && !loading ? 'linear-gradient(135deg, #12b863 0%, #0d8c4a 100%)' : 'rgba(18, 184, 99, 0.15)',
                    color: file && !loading ? '#fff' : '#4d695c',
                    cursor: file && !loading ? 'pointer' : 'not-allowed',
                    border: 'none',
                    padding: '16px',
                    borderRadius: '8px',
                    fontWeight: 600,
                    transition: '0.3s',
                    boxShadow: file && !loading ? '0 4px 15px rgba(18, 184, 99, 0.25)' : 'none'
                  }}
                >
                  {loading ? 'Evaluating Model...' : 'Start Classification'}
                </button>
              </div>
            </div>

          </div>

          {/* Loader Overlay */}
          {loading && (
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(3, 10, 6, 0.85)',
              zIndex: 5,
              borderRadius: '16px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              padding: '24px'
            }}>
              <div className="wave-loader" style={{ marginBottom: '24px' }}>
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
                <div className="wave-bar"></div>
              </div>
              <h4 style={{ color: '#fff', fontSize: '1.2rem', marginBottom: '8px', textShadow: '0 0 10px rgba(18,184,99,0.3)' }}>Analyzing Soundscape</h4>
              <p style={{ color: '#8ea79b', fontSize: '0.9rem', fontStyle: 'italic' }}>{loadingStage}</p>
            </div>
          )}
        </div>

        {/* Results Section */}
        {results && (
          <div className="glass" style={{ padding: '32px', marginBottom: '64px' }}>

            {/* Tabs for result visualization */}
            <div className="tabs-container">
              <button
                className={`tab-btn ${activeTab === 'detections' ? 'active' : ''}`}
                onClick={() => setActiveTab('detections')}
              >
                Identified Species
              </button>
              <button
                className={`tab-btn ${activeTab === 'taxamoe' ? 'active' : ''}`}
                onClick={() => setActiveTab('taxamoe')}
              >
                TaxaMoE Router Weights
              </button>
              <button
                className={`tab-btn ${activeTab === 'spectrogram' ? 'active' : ''}`}
                onClick={() => setActiveTab('spectrogram')}
              >
                Mel Spectrogram
              </button>
            </div>

            {/* TAB CONTENT: DETECTIONS */}
            {activeTab === 'detections' && (
              <div>
                <h3 className="card-title">Top Identified Bioacoustic Signals</h3>

                <div className="glass" style={{
                  padding: '12px 16px',
                  borderRadius: '8px',
                  background: 'rgba(212, 175, 55, 0.05)',
                  border: '1px solid rgba(212, 175, 55, 0.2)',
                  marginBottom: '20px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px'
                }}>
                  <span style={{ fontSize: '1.2rem' }}>ℹ️</span>
                  <p style={{ fontSize: '0.85rem', color: '#c2dfcf', margin: 0, lineHeight: 1.4 }}>
                    <strong>Note on Scores:</strong> The model uses <strong>Rank-based Prior-Axis Fusion</strong> to integrate SED, ProtoSSM, and Perch models. Scores represent the peak rank percentile across audio windows. Highly prominent species naturally reach 100% confidence, indicating maximum signal ranking in their windows.
                  </p>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '20px' }}>
                  {results.detections.map((det, index) => {
                    const isHighConfidence = det.score > 0.65;
                    const isMidConfidence = det.score <= 0.65 && det.score > 0.4;

                    let confidenceColor = '#12b863'; // Green
                    if (isMidConfidence) confidenceColor = '#d4af37'; // Gold
                    else if (det.score <= 0.4) confidenceColor = '#ff6b6b'; // Red

                    return (
                      <div
                        key={index}
                        style={{
                          padding: '20px',
                          background: 'rgba(3, 15, 8, 0.4)',
                          border: '1px solid rgba(18, 184, 99, 0.1)',
                          borderRadius: '12px',
                          display: 'flex',
                          flexDirection: 'column',
                          justifyContent: 'space-between',
                          position: 'relative',
                          overflow: 'hidden'
                        }}
                      >
                        {/* Glow indicator */}
                        <div style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '4px',
                          height: '100%',
                          backgroundColor: confidenceColor
                        }} />

                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                            <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#8ea79b', fontWeight: 600 }}>{det.class_name}</span>
                            <span style={{ fontSize: '0.85rem', fontWeight: 700, color: confidenceColor }}>{(det.score * 100).toFixed(1)}%</span>
                          </div>

                          <h4 style={{ fontSize: '1.2rem', color: '#fff', marginBottom: '4px', fontFamily: 'var(--font-sans)', fontWeight: 600 }}>{det.label}</h4>
                          <p style={{ fontSize: '0.85rem', color: '#8ea79b', fontStyle: 'italic', marginBottom: '16px' }}>{det.sci_name}</p>
                        </div>

                        {/* Progress bar */}
                        <div style={{ height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{
                            width: `${det.score * 100}%`,
                            height: '100%',
                            backgroundColor: confidenceColor,
                            boxShadow: `0 0 8px ${confidenceColor}`
                          }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* TAB CONTENT: TAXAMOE ROUTER */}
            {activeTab === 'taxamoe' && (
              <div>
                <h3 className="card-title">Taxa-aware Mixture of Experts Routing</h3>
                <p style={{ color: '#8ea79b', fontSize: '0.9rem', marginBottom: '24px', maxWidth: '800px' }}>
                  TaxaMoE routes embeddings dynamically to specialized prototypical heads. Connection lines brighten based on the gating weights allocated by the routing network.
                </p>

                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  padding: '40px 0',
                  backgroundImage: 'radial-gradient(circle, rgba(18, 184, 99, 0.05) 0%, transparent 70%)',
                  borderRadius: '16px'
                }}>
                  {/* Central Router Node */}
                  <div style={{
                    width: '120px',
                    height: '120px',
                    borderRadius: '60px',
                    background: 'radial-gradient(circle, #0e4c27 0%, #030a06 100%)',
                    border: '2px solid #12b863',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    alignItems: 'center',
                    boxShadow: '0 0 30px rgba(18, 184, 99, 0.3)',
                    zIndex: 2,
                    marginBottom: '40px'
                  }}>
                    <span style={{ fontSize: '1.8rem' }}>🧠</span>
                    <span style={{ fontSize: '0.8rem', color: '#fff', fontWeight: 'bold', marginTop: '4px' }}>TaxaMoE</span>
                    <span style={{ fontSize: '0.65rem', color: '#8ea79b' }}>Router</span>
                  </div>

                  {/* Pathways Grid */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '16px', width: '100%', maxWidth: '900px' }}>
                    {Object.entries(results.router_weights).map(([taxName, weight]) => {
                      const glowIntensity = Math.max(weight, 0.05);
                      const activeColor = `rgba(18, 184, 99, ${glowIntensity})`;
                      const borderGlow = `0 0 ${glowIntensity * 25}px rgba(18, 184, 99, ${glowIntensity * 0.5})`;

                      const icon = {
                        Aves: '🦅',
                        Insecta: '🦟',
                        Amphibia: '🐸',
                        Mammalia: '🐆',
                        Reptilia: '🐍'
                      }[taxName] || '🐾';

                      return (
                        <div
                          key={taxName}
                          style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            padding: '20px 10px',
                            background: 'rgba(8, 26, 17, 0.45)',
                            border: `1px solid ${weight > 0.3 ? 'rgba(18, 184, 99, 0.4)' : 'rgba(18, 184, 99, 0.1)'}`,
                            boxShadow: weight > 0.3 ? borderGlow : 'none',
                            borderRadius: '12px',
                            transition: 'all 0.3s ease'
                          }}
                        >
                          <span style={{ fontSize: '2rem', marginBottom: '8px' }}>{icon}</span>
                          <span style={{ color: '#fff', fontWeight: 600, fontSize: '1rem', marginBottom: '4px' }}>{taxName}</span>
                          <span style={{ color: '#12b863', fontWeight: 700, fontSize: '1.1rem' }}>{(weight * 100).toFixed(1)}%</span>

                          {/* Mini Progress track */}
                          <div style={{ width: '80%', height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', marginTop: '12px', overflow: 'hidden' }}>
                            <div style={{ width: `${weight * 100}%`, height: '100%', backgroundColor: '#12b863' }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* TAB CONTENT: SPECTROGRAM */}
            {activeTab === 'spectrogram' && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <div>
                    <h3 className="card-title" style={{ marginBottom: '4px' }}>Interactive Mel Spectrogram</h3>
                    <p style={{ color: '#8ea79b', fontSize: '0.85rem' }}>Visualizes frequency content of the first 10 seconds of soundscape. Play the audio to track the cursor.</p>
                  </div>

                  {audioUrl && (
                    <button
                      onClick={handlePlayPause}
                      style={{
                        background: '#d4af37',
                        border: 'none',
                        color: '#030a06',
                        padding: '10px 24px',
                        borderRadius: '30px',
                        fontWeight: 'bold',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        boxShadow: '0 4px 15px rgba(212, 175, 55, 0.25)',
                        transition: '0.3s'
                      }}
                    >
                      <span>{isPlaying ? '⏸ Pause' : '▶ Play'}</span>
                    </button>
                  )}
                </div>

                {audioUrl && (
                  <audio
                    ref={audioRef}
                    src={audioUrl}
                    onEnded={handleAudioEnded}
                    style={{ display: 'none' }}
                  />
                )}

                <canvas
                  ref={canvasRef}
                  className="spec-canvas"
                  style={{ width: '100%', height: '280px', marginBottom: '12px' }}
                />

                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#8ea79b', fontSize: '0.8rem' }}>
                  <span>0s</span>
                  <span>Playback: {playbackTime.toFixed(1)}s / 10.0s</span>
                  <span>10.0s (First window clip)</span>
                </div>
              </div>
            )}

          </div>
        )}

        {/* Model Architecture details */}
        <section id="about" style={{ padding: '40px 0', borderTop: '1px solid rgba(18, 184, 99, 0.1)' }}>
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '2.2rem', color: '#fff', textAlign: 'center', marginBottom: '48px' }}>
            TaxaMoE Ensemble Architecture
          </h2>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '48px', alignItems: 'center' }}>

            {/* Diagram */}
            <div className="glass" style={{ padding: '32px', border: '1px solid rgba(18, 184, 99, 0.15)' }}>
              <h3 style={{ fontFamily: 'var(--font-serif)', color: '#d4af37', fontSize: '1.4rem', marginBottom: '24px' }}>Pipeline Topology</h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ padding: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(18,184,99,0.1)', borderRadius: '8px' }}>
                  <strong style={{ color: '#fff' }}>1. Audio Chunking:</strong>
                  <p style={{ fontSize: '0.85rem', color: '#8ea79b' }}>60s soundscape is split into 12 5-second windows.</p>
                </div>

                <div style={{ padding: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(18,184,99,0.1)', borderRadius: '8px' }}>
                  <strong style={{ color: '#fff' }}>2. Feature Extraction (Perch v2):</strong>
                  <p style={{ fontSize: '0.85rem', color: '#8ea79b' }}>Generates 1536-dimensional embeddings for each window.</p>
                </div>

                <div style={{ padding: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(18,184,99,0.1)', borderRadius: '8px' }}>
                  <strong style={{ color: '#fff' }}>3. TaxaMoE Routing:</strong>
                  <p style={{ fontSize: '0.85rem', color: '#8ea79b' }}>Routes coordinates to Aves, Insecta, Amphibia, Mammalia, Reptilia experts.</p>
                </div>

                <div style={{ padding: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(18,184,99,0.1)', borderRadius: '8px' }}>
                  <strong style={{ color: '#fff' }}>4. Ensembling & Calibration:</strong>
                  <p style={{ fontSize: '0.85rem', color: '#8ea79b' }}>Rank blending of SED, ProtoSSM, and Perch, followed by Bayesian site/hour prior adjustment.</p>
                </div>
              </div>
            </div>

            {/* Explanation text */}
            <div>
              <h3 style={{ fontFamily: 'var(--font-serif)', color: '#fff', fontSize: '1.8rem', marginBottom: '16px' }}>Correcting Taxa Distribution Mismatch</h3>
              <p style={{ color: '#8ea79b', marginBottom: '20px', fontSize: '0.95rem', lineHeight: 1.7 }}>
                Passive acoustic monitoring (PAM) datasets are notorious for species bias. The training set is <strong>97.9% bird vocalizations</strong>, whereas the actual test soundscapes from the Pantanal wetlands contain over <strong>30% non-bird classes</strong> (frogs, insects, reptiles, and mammals).
              </p>
              <p style={{ color: '#8ea79b', marginBottom: '20px', fontSize: '0.95rem', lineHeight: 1.7 }}>
                Under standard models, non-bird calls are frequently ignored or misclassified as birds due to the extreme class imbalance. Our <strong>Taxa-aware Mixture of Experts (TaxaMoE)</strong> architecture solves this by adapter-projecting the feature space and routing species to specialized prototypical heads trained exclusively on their taxonomical groups.
              </p>
              <p style={{ color: '#8ea79b', fontSize: '0.95rem', lineHeight: 1.7 }}>
                This is combined with a <strong>Selective State Space Model (ProtoSSM)</strong> to capture long-range temporal structures across the audio windows, and <strong>Sound Event Detection (SED)</strong> to locate the exact frame-level boundaries of the call.
              </p>
            </div>

          </div>
        </section>

        {/* Kaggle Competition Achievements section */}
        <section id="kaggle" style={{ padding: '80px 0', borderTop: '1px solid rgba(18, 184, 99, 0.15)' }}>
          <div style={{ textAlign: 'center', marginBottom: '40px' }}>
            <span style={{ color: '#d4af37', textTransform: 'uppercase', fontSize: '0.85rem', fontWeight: 600, letterSpacing: '3px', display: 'block', marginBottom: '8px' }}>COMPETITION CASE STUDY</span>
            <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '2.2rem', color: '#fff', marginBottom: '12px' }}>
              BirdCLEF+ 2026 Kaggle Challenge
            </h2>
            <p style={{ color: '#8ea79b', maxWidth: '700px', margin: '0 auto' }}>
              This platform runs the exact model ensemble designed for the Cornell Lab of Ornithology&apos;s Research Code Competition on Kaggle.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '48px', alignItems: 'center' }}>
            {/* Left: Competition Overview & Rules */}
            <div>
              <h3 style={{ fontFamily: 'var(--font-serif)', color: '#12b863', fontSize: '1.4rem', marginBottom: '16px' }}>The Task: Acoustic Species Identification</h3>
              <p style={{ color: '#8ea79b', fontSize: '0.95rem', lineHeight: 1.7, marginBottom: '16px' }}>
                How do you protect an ecosystem you cannot fully see? You listen. The BirdCLEF+ 2026 competition challenged participants to develop machine learning frameworks capable of identifying understudied species within continuous, messy, field-collected passive acoustic monitoring (PAM) data from the Pantanal wetlands.
              </p>
              <p style={{ color: '#8ea79b', fontSize: '0.95rem', lineHeight: 1.7, marginBottom: '24px' }}>
                Organized by the <strong>Cornell Lab of Ornithology</strong>, <strong>Google DeepMind</strong>, and <strong>iNaturalist</strong>, the contest drew global ML researchers to build models capable of generalizing across different micro-habitats and seasons, evaluated using a macro-averaged ROC-AUC metric.
              </p>

              <h4 style={{ color: '#fff', fontSize: '1rem', marginBottom: '12px', fontWeight: 600 }}>Competition Notebooks:</h4>
              <ul style={{ listStyle: 'none', padding: 0, margin: '0 0 24px 0', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <li>
                  <a href="https://www.kaggle.com/code/kritikabenjwal/birdclef26-inference-submission" target="_blank" rel="noopener noreferrer" style={{ color: '#d4af37', textDecoration: 'none', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '8px' }} className="hover-glow">
                    <span>📓</span> <span>BirdCLEF+ 2026 Inference Submission Notebook</span>
                  </a>
                </li>
                <li>
                  <a href="https://www.kaggle.com/code/kritikabenjwal/birdclef26-taxamoe-training" target="_blank" rel="noopener noreferrer" style={{ color: '#d4af37', textDecoration: 'none', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '8px' }} className="hover-glow">
                    <span>📓</span> <span>TaxaMoE Model Training Pipeline Notebook</span>
                  </a>
                </li>
              </ul>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                <div className="glass" style={{ padding: '16px', textAlign: 'center' }}>
                  <span style={{ fontSize: '1.8rem', display: 'block', marginBottom: '4px' }}>👥</span>
                  <strong style={{ color: '#fff', fontSize: '1.2rem', display: 'block' }}>5,243</strong>
                  <span style={{ fontSize: '0.75rem', color: '#8ea79b' }}>Participants</span>
                </div>
                <div className="glass" style={{ padding: '16px', textAlign: 'center' }}>
                  <span style={{ fontSize: '1.8rem', display: 'block', marginBottom: '4px' }}>🤝</span>
                  <strong style={{ color: '#fff', fontSize: '1.2rem', display: 'block' }}>4,243</strong>
                  <span style={{ fontSize: '0.75rem', color: '#8ea79b' }}>Teams</span>
                </div>
                <div className="glass" style={{ padding: '16px', textAlign: 'center' }}>
                  <span style={{ fontSize: '1.8rem', display: 'block', marginBottom: '4px' }}>📊</span>
                  <strong style={{ color: '#fff', fontSize: '1.2rem', display: 'block' }}>159,064</strong>
                  <span style={{ fontSize: '0.75rem', color: '#8ea79b' }}>Submissions</span>
                </div>
              </div>
            </div>

            {/* Right: Leaderboard Card celebrating user */}
            <div className="glass" style={{
              padding: '32px',
              border: '1px solid rgba(212, 175, 55, 0.3)',
              boxShadow: '0 0 30px rgba(212, 175, 55, 0.05)',
              position: 'relative',
              overflow: 'hidden',
              borderRadius: '20px',
              background: 'linear-gradient(135deg, rgba(8, 26, 17, 0.4) 0%, rgba(3, 10, 6, 0.6) 100%)'
            }}>
              {/* Gold glowing accent */}
              <div style={{
                position: 'absolute',
                top: 0,
                right: 0,
                width: '60px',
                height: '60px',
                background: 'radial-gradient(circle, rgba(212,175,55,0.2) 0%, transparent 70%)',
                pointerEvents: 'none'
              }} />

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '24px' }}>
                <span style={{ fontSize: '1.5rem' }}>🏆</span>
                <span style={{ color: '#d4af37', fontWeight: 600, fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '2px' }}>Leaderboard Standing</span>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <span style={{ fontSize: '0.8rem', color: '#8ea79b', display: 'block', marginBottom: '4px' }}>Competitor</span>
                <span style={{ fontSize: '1.4rem', color: '#fff', fontWeight: 'bold' }}>Kritika Benjwal</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px', borderTop: '1px solid rgba(255, 255, 255, 0.05)', paddingTop: '16px' }}>
                <div>
                  <span style={{ fontSize: '0.8rem', color: '#8ea79b', display: 'block', marginBottom: '4px' }}>Score (ROC-AUC)</span>
                  <span style={{ fontSize: '1.6rem', color: '#12b863', fontWeight: 'bold' }}>0.93957</span>
                </div>
                <div>
                  <span style={{ fontSize: '0.8rem', color: '#8ea79b', display: 'block', marginBottom: '4px' }}>Final Rank</span>
                  <span style={{ fontSize: '1.6rem', color: '#d4af37', fontWeight: 'bold' }}>1621 / 4243</span>
                </div>
              </div>

              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <span style={{ fontSize: '0.75rem', color: '#8ea79b', display: 'block', marginBottom: '4px', fontFamily: 'monospace' }}>SUBMISSION METADATA</span>
                <p style={{ fontSize: '0.8rem', color: '#c2dfcf', margin: 0, lineHeight: 1.4 }}>
                  Ensembled Perch embeddings, TaxaMoE gating matrices, ProtoSSM sequences, and sound event detectors combined to bypass wetland taxonomic imbalance.
                </p>
              </div>
            </div>
          </div>
        </section>

      </main>

      {/* Footer */}
      <footer style={{
        marginTop: 'auto',
        borderTop: '1px solid rgba(18, 184, 99, 0.15)',
        padding: '48px 24px',
        backgroundColor: '#020804',
        position: 'relative',
        zIndex: 2
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px', alignItems: 'center' }}>

          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '24px', fontSize: '0.9rem' }}>
            <a href="https://www.linkedin.com/in/kritika-benjwal" target="_blank" rel="noopener noreferrer" style={{ color: '#8ea79b', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '6px' }} className="hover-glow">
              🔗 LinkedIn
            </a>
            <a href="https://github.com/Kritika11052005" target="_blank" rel="noopener noreferrer" style={{ color: '#8ea79b', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '6px' }} className="hover-glow">
              🐙 GitHub
            </a>
            <a href="mailto:ananya.benjwal@gmail.com" style={{ color: '#8ea79b', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '6px' }} className="hover-glow">
              📧 ananya.benjwal@gmail.com
            </a>
            <a href="https://www.kaggle.com/kritikabenjwal" target="_blank" rel="noopener noreferrer" style={{ color: '#8ea79b', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '6px' }} className="hover-glow">
              🏆 Kaggle Profile
            </a>
            <a href="https://huggingface.co/spaces/Kritzzz11/bio-acoustica" target="_blank" rel="noopener noreferrer" style={{ color: '#8ea79b', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '6px' }} className="hover-glow">
              🤗 Hugging Face Space
            </a>
          </div>

          <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', width: '100%', paddingTop: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
            <span style={{ fontSize: '0.85rem', color: '#8ea79b' }}>
              © 2026 WildEcho platform. Designed and built by <strong>Kritika Benjwal</strong>.
            </span>
            <span style={{ fontSize: '0.85rem', color: '#8ea79b' }}>
              Powered by Next.js & FastAPI backend.
            </span>
          </div>

        </div>
      </footer>

    </div>
  );
}
