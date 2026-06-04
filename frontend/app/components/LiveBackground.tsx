'use client';

import React, { useEffect, useRef } from 'react';

interface Particle {
  x: number;
  y: number;
  size: number;
  baseSpeedX: number;
  baseSpeedY: number;
  speedX: number;
  speedY: number;
  alpha: number;
  alphaSpeed: number;
  color: string;
  angle: number;
  angleSpeed: number;
}

export default function LiveBackground() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const particlesRef = useRef<Particle[]>([]);
  const mouseRef = useRef<{ x: number | null; y: number | null }>({ x: null, y: null });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;

    const resizeCanvas = () => {
      if (!canvas) return;
      const parent = canvas.parentElement;
      canvas.width = parent?.clientWidth || window.innerWidth;
      canvas.height = parent?.clientHeight || window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Initialize fireflies
    const maxParticles = 60;
    const particles: Particle[] = [];
    const colors = [
      'rgba(18, 184, 99, ', // Green
      'rgba(212, 175, 55, ', // Gold
      'rgba(226, 240, 232, ', // Light green-white
    ];

    for (let i = 0; i < maxParticles; i++) {
      const size = Math.random() * 2.5 + 0.8;
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size,
        baseSpeedX: (Math.random() - 0.5) * 0.4,
        baseSpeedY: (Math.random() - 0.5) * 0.4 - 0.15, // float slightly upward
        speedX: 0,
        speedY: 0,
        alpha: Math.random() * 0.6 + 0.1,
        alphaSpeed: (Math.random() * 0.005 + 0.002) * (Math.random() > 0.5 ? 1 : -1),
        color: colors[Math.floor(Math.random() * colors.length)],
        angle: Math.random() * Math.PI * 2,
        angleSpeed: Math.random() * 0.02 - 0.01,
      });
    }
    particlesRef.current = particles;

    // Track mouse moves to create interactive sparks
    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      // Only record mouse position if it's within the hero section bounds
      if (x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height) {
        mouseRef.current = { x, y };
        
        // Spawn active sparks around mouse occasionally
        if (Math.random() < 0.35) {
          spawnSpark(x, y);
        }
      } else {
        mouseRef.current = { x: null, y: null };
      }
    };

    const handleMouseLeave = () => {
      mouseRef.current = { x: null, y: null };
    };

    window.addEventListener('mousemove', handleMouseMove);
    canvas.parentElement?.addEventListener('mouseleave', handleMouseLeave);

    const spawnSpark = (x: number, y: number) => {
      const parts = particlesRef.current;
      if (parts.length > 100) return; // Cap maximum active particles

      parts.push({
        x,
        y,
        size: Math.random() * 1.5 + 0.5,
        baseSpeedX: (Math.random() - 0.5) * 1.2,
        baseSpeedY: -Math.random() * 1.2 - 0.5, // float upward faster
        speedX: 0,
        speedY: 0,
        alpha: 1.0,
        alphaSpeed: -0.015 - Math.random() * 0.01, // fade out quickly
        color: 'rgba(212, 175, 55, ', // Gold sparks
        angle: Math.random() * Math.PI * 2,
        angleSpeed: Math.random() * 0.1 - 0.05,
      });
    };

    // Main animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const parts = particlesRef.current;
      const mouse = mouseRef.current;

      for (let i = parts.length - 1; i >= 0; i--) {
        const p = parts[i];

        // Fade in/out logic
        p.alpha += p.alphaSpeed;
        if (p.alpha > 0.8) {
          p.alpha = 0.8;
          p.alphaSpeed = -p.alphaSpeed;
        } else if (p.alpha < 0.05) {
          // If a natural firefly fades out completely, reset it randomly
          if (p.alphaSpeed < 0 && p.alphaSpeed > -0.01) {
            p.x = Math.random() * canvas.width;
            p.y = Math.random() * canvas.height;
            p.alpha = 0.05;
            p.alphaSpeed = Math.random() * 0.005 + 0.002;
          } else {
            // Remove short-lived mouse sparks
            parts.splice(i, 1);
            continue;
          }
        }

        // Add wave movement to firefly motion
        p.angle += p.angleSpeed;
        p.speedX = p.baseSpeedX + Math.sin(p.angle) * 0.15;
        p.speedY = p.baseSpeedY + Math.cos(p.angle) * 0.1;

        // Attract particles slightly towards mouse if close
        if (mouse.x !== null && mouse.y !== null) {
          const dx = mouse.x - p.x;
          const dy = mouse.y - p.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120 && dist > 0.1) {
            // Apply slight pull vector
            p.speedX += (dx / dist) * 0.08;
            p.speedY += (dy / dist) * 0.08;
          }
        }

        p.x += p.speedX;
        p.y += p.speedY;

        // Wrap around boundaries for infinite loop
        if (p.x < -10) p.x = canvas.width + 10;
        if (p.x > canvas.width + 10) p.x = -10;
        if (p.y < -10) p.y = canvas.height + 10;
        if (p.y > canvas.height + 10) p.y = -10;

        // Safety check to ensure coordinates are finite
        if (!isFinite(p.x) || !isFinite(p.y)) {
          p.x = Math.random() * canvas.width;
          p.y = Math.random() * canvas.height;
          p.speedX = p.baseSpeedX;
          p.speedY = p.baseSpeedY;
        }

        // Render particle with glow
        ctx.beginPath();
        
        // Draw glow
        const glowRadius = p.size * 3.5;
        if (glowRadius > 0 && isFinite(glowRadius)) {
          try {
            const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowRadius);
            grad.addColorStop(0, p.color + p.alpha + ')');
            grad.addColorStop(0.3, p.color + p.alpha * 0.5 + ')');
            grad.addColorStop(1, 'rgba(0,0,0,0)');
            
            ctx.fillStyle = grad;
            ctx.arc(p.x, p.y, glowRadius, 0, Math.PI * 2);
            ctx.fill();
          } catch (e) {
            // Silently catch and recover from draw errors
          }
        }

        // Draw center bright point
        ctx.beginPath();
        ctx.fillStyle = `rgba(255, 255, 255, ${p.alpha})`;
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return <canvas ref={canvasRef} className="fireflies-canvas" />;
}
