/**
 * CQTAnalyzer.js
 * A production-ready, high-precision visualizer with logarithmic frequency mapping,
 * custom precision grids, and 432 Hz harmonic snap detection.
 */
class CQTAnalyzer {
    constructor(audioContext, options = {}) {
        this.ctx = audioContext;
        
        // Configuration
        this.fftSize = options.fftSize || 16384; // High resolution for precise low-mid tuning
        this.minFreq = options.minFreq || 20;
        this.maxFreq = options.maxFreq || 2000;
        this.snapTolerance = options.snapTolerance || 2.0; // ± Hz to trigger a snap visual
        
        // Create internal Web Audio Analyser
        this.analyser = this.ctx.createAnalyser();
        this.analyser.fftSize = this.fftSize;
        this.analyser.smoothingTimeConstant = options.smoothing || 0.75;
        
        this.bufferLength = this.analyser.frequencyBinCount;
        this.dataArray = new Uint8Array(this.bufferLength);
        
        // Targeted Grid Lines (Solfeggio + Cosmic Harmonics)
        this.gridFrequencies = options.gridFrequencies || [
            27, 54, 174, 285, 396, 417, 432, 528, 639, 741, 852, 963, 1296, 1740
        ];
        
        // Specific 432Hz Harmonics array for secondary overlay tracking
        this.harmonics432 = [432, 864, 1296, 1728];
        
        // Canvas tracking
        this.canvas = null;
        this.canvasCtx = null;
        this.animationFrameId = null;
    }

    /**
     * Connect the analyzer to any source AudioNode (e.g., your Master Gain node)
     * @param {AudioNode} sourceNode 
     */
    connectSource(sourceNode) {
        sourceNode.connect(this.analyser);
    }

    /**
     * Bind the renderer to an HTML5 Canvas element
     * @param {HTMLCanvasElement} canvasElement 
     */
    bindCanvas(canvasElement) {
        this.canvas = canvasElement;
        this.canvasCtx = this.canvas.getContext('2d');
        this.resizeCanvas();
        
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    resizeCanvas() {
        if (!this.canvas) return;
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.canvasCtx.scale(dpr, dpr);
    }

    /**
     * Converts a frequency value to its corresponding X coordinate on a logarithmic scale
     */
    freqToX(freq, width) {
        const logMin = Math.log10(this.minFreq);
        const logMax = Math.log10(this.maxFreq);
        const logFreq = Math.log10(freq);
        return ((logFreq - logMin) / (logMax - logMin)) * width;
    }

    /**
     * Main animation loop wrapper
     */
    start() {
        if (!this.canvasCtx) {
            console.error("CQTAnalyzer: No canvas bound. Call bindCanvas() first.");
            return;
        }
        const drawLoop = () => {
            this.animationFrameId = requestAnimationFrame(drawLoop);
            this.render();
        };
        drawLoop();
    }

    stop() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
    }

    /**
     * Core Visualizer Rendering Engine
     */
    render() {
        const width = this.canvas.width / (window.devicePixelRatio || 1);
        const height = this.canvas.height / (window.devicePixelRatio || 1);
        const ctx = this.canvasCtx;

        // 1. Fetch real-time frequency data
        this.analyser.getByteFrequencyData(this.dataArray);

        // 2. Clear background with deep space dark tone
        ctx.fillStyle = '#0a0b10';
        ctx.fillRect(0, 0, width, height);

        // 3. Draw Logarithmic CQT Spectrum Graph
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgba(0, 233, 255, 0.85)'; // Cyber cyan spectrum
        ctx.beginPath();

        const sampleRate = this.ctx.sampleRate;
        let firstPoint = true;

        for (let i = 0; i < this.bufferLength; i++) {
            // Calculate actual frequency of this FFT bin
            const freq = (i * sampleRate) / this.fftSize;
            if (freq < this.minFreq || freq > this.maxFreq) continue;

            const x = this.freqToX(freq, width);
            // Normalize amplitude to height percentage
            const amplitude = this.dataArray[i] / 255;
            const y = height - (amplitude * height * 0.8) - 30; // Leave space for labels at bottom

            if (firstPoint) {
                ctx.moveTo(x, y);
                firstPoint = false;
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // 4. Find Peak Frequencies to calculate Snaps
        const peaks = [];
        const threshold = 120; // Ignore low noise floor values (0-255)
        for (let i = 1; i < this.bufferLength - 1; i++) {
            if (this.dataArray[i] > this.dataArray[i-1] && this.dataArray[i] > this.dataArray[i+1]) {
                if (this.dataArray[i] > threshold) {
                    const exactFreq = (i * sampleRate) / this.fftSize;
                    peaks.push({ freq: exactFreq, val: this.dataArray[i] });
                }
            }
        }

        // 5. Draw Precision Grid Layer & Process Overlay Hooks
        this.gridFrequencies.forEach(targetFreq => {
            if (targetFreq < this.minFreq || targetFreq > this.maxFreq) return;

            const x = this.freqToX(targetFreq, width);
            
            // Check if any current audio peak matches this grid line within our ± snap tolerance
            const isSnapped = peaks.some(p => Math.abs(p.freq - targetFreq) <= this.snapTolerance);
            const isCore432 = this.harmonics432.includes(targetFreq);

            // Styling adjustments based on state and meaning
            if (isSnapped) {
                ctx.strokeStyle = isCore432 ? '#ffcc00' : '#00ff88'; // Gold for 432 family snap, neon green for others
                ctx.lineWidth = 2.5;
                ctx.shadowBlur = 10;
                ctx.shadowColor = ctx.strokeStyle;
            } else {
                ctx.strokeStyle = isCore432 ? 'rgba(255, 204, 0, 0.25)' : 'rgba(255, 255, 255, 0.1)';
                ctx.lineWidth = 1;
                ctx.shadowBlur = 0;
            }

            // Draw line
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height - 25);
            ctx.stroke();
            ctx.shadowBlur = 0; // Reset blur

            // Draw small clean text labels
            ctx.fillStyle = isSnapped ? '#ffffff' : 'rgba(255, 255, 255, 0.4)';
            ctx.font = isSnapped ? 'bold 10px monospace' : '10px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`${targetFreq}Hz`, x, height - 10);
        });

        // 6. Secondary UI Component: Top Corner 432 Hz Lock Indicator
        const main432Lock = peaks.some(p => Math.abs(p.freq - 432) <= this.snapTolerance);
        if (main432Lock) {
            ctx.fillStyle = 'rgba(255, 204, 0, 0.15)';
            ctx.fillRect(10, 10, 150, 30);
            ctx.strokeStyle = '#ffcc00';
            ctx.strokeRect(10, 10, 150, 30);
            
            ctx.fillStyle = '#ffcc00';
            ctx.font = 'bold 11px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText('✨ 432 Hz HARMONIC LOCK', 20, 28);
        }
    }
}

// Export module for native JS / Electron environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CQTAnalyzer;
} else {
    window.CQTAnalyzer = CQTAnalyzer;
}