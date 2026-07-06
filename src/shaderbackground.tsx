import { useEffect, useRef } from 'react';
import { Renderer, Program, Color, Mesh, Triangle } from 'ogl';

const vertexShader = /* glsl */ `
  attribute vec2 uv;
  attribute vec2 position;
  varying vec2 vUv;
  void main() {
      vUv = uv;
      gl_Position = vec4(position, 0, 1);
  }
`;

// Dark cinematic AI theme fragment shader structure
const fragmentShader = /* glsl */ `
  precision highp float;
  uniform float uTime;
  uniform vec3 uColor;
  varying vec2 vUv;
  void main() {
      // Lowered multiplier for darker aesthetic
      gl_FragColor.rgb = 0.1 + 0.15 * cos(vUv.xyx + uTime) + uColor;
      gl_FragColor.a = 1.0;
  }
`;

export const ShaderBackground = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const renderer = new Renderer();
    const gl = renderer.gl;
    containerRef.current.appendChild(gl.canvas);
    gl.clearColor(0, 0, 0, 1);

    const handleResize = () => {
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener('resize', handleResize, false);
    handleResize();

    const geometry = new Triangle(gl);
    const program = new Program(gl, {
      vertex: vertexShader,
      fragment: fragmentShader,
      uniforms: {
        uTime: { value: 0 },
        // Deep black / dark purple base color
        uColor: { value: new Color(0.1, 0.05, 0.25) },
      },
    });

    const mesh = new Mesh(gl, { geometry, program });

    let animationId: number;
    const update = (t: number) => {
      animationId = requestAnimationFrame(update);
      program.uniforms.uTime.value = t * 0.001;
      renderer.render({ scene: mesh });
    };
    animationId = requestAnimationFrame(update);

    // Memory management: prevent canvas leaks on unconditional unmounts
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationId);
      if (containerRef.current && gl.canvas) {
        containerRef.current.removeChild(gl.canvas);
      }
      const ext = gl.getExtension('WEBGL_lose_context');
      if (ext) ext.loseContext();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100vw',
        height: '100vh',
        position: 'absolute',
        top: 0,
        left: 0,
        pointerEvents: 'none',
      }}
    />
  );
};