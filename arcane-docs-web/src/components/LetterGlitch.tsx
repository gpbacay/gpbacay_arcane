import { useRef, useEffect } from 'react';

interface Letter {
  char: string;
  color: string;
  targetColor: string;
  colorProgress: number;
  isHighlighted?: boolean;
}

interface RGBColor {
  r: number;
  g: number;
  b: number;
}

const LetterGlitch = ({
  glitchColors = ['#B9DFE0', '#F294C0', '#C785F2', '#835BD9', '#9DE4FA'],
  className = '',
  glitchSpeed = 50,
  centerVignette = true,
  outerVignette = false,
  smooth = true,
  characters = '010101010101                     ARCANEARCANEARCANEARCANEARCANEARCANEARCANEARCANE'
}: {
  glitchColors?: string[];
  className?: string;
  glitchSpeed?: number;
  centerVignette?: boolean;
  outerVignette?: boolean;
  smooth?: boolean;
  characters?: string;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const letters = useRef<Letter[]>([]);
  const grid = useRef({ columns: 0, rows: 0 });
  const context = useRef<CanvasRenderingContext2D | null>(null);
  const lastGlitchTime = useRef<number>(Date.now());

  const lettersAndSymbols = Array.from(characters);

  const fontSize = 16;
  const charWidth = 10;
  const charHeight = 20;

  const getRandomChar = () => {
    return lettersAndSymbols[Math.floor(Math.random() * lettersAndSymbols.length)];
  };

  const detectAndHighlightArcane = () => {
    if (!letters.current || letters.current.length === 0) return;
    
    const targetWord = 'ARCANE';
    const wordLength = targetWord.length;
    
    // Reset all highlights
    letters.current.forEach(letter => {
      letter.isHighlighted = false;
    });
    
    // Check horizontal sequences
    for (let row = 0; row < grid.current.rows; row++) {
      for (let col = 0; col <= grid.current.columns - wordLength; col++) {
        let found = true;
        const indices = [];
        
        for (let i = 0; i < wordLength; i++) {
          const index = row * grid.current.columns + (col + i);
          if (index < letters.current.length && letters.current[index].char === targetWord[i]) {
            indices.push(index);
          } else {
            found = false;
            break;
          }
        }
        
        if (found) {
          indices.forEach(idx => {
            letters.current[idx].isHighlighted = true;
          });
        }
      }
    }
    
    // Check vertical sequences
    for (let col = 0; col < grid.current.columns; col++) {
      for (let row = 0; row <= grid.current.rows - wordLength; row++) {
        let found = true;
        const indices = [];
        
        for (let i = 0; i < wordLength; i++) {
          const index = (row + i) * grid.current.columns + col;
          if (index < letters.current.length && letters.current[index].char === targetWord[i]) {
            indices.push(index);
          } else {
            found = false;
            break;
          }
        }
        
        if (found) {
          indices.forEach(idx => {
            letters.current[idx].isHighlighted = true;
          });
        }
      }
    }
  };

  const getRandomColor = () => {
    const hexColor = glitchColors[Math.floor(Math.random() * glitchColors.length)];
    const rgbColor = hexToRgb(hexColor);
    if (rgbColor) {
      return `rgba(${rgbColor.r}, ${rgbColor.g}, ${rgbColor.b}, 0.3)`;
    }
    return hexColor; // fallback to original if conversion fails
  };

  const hexToRgb = (hex: string) => {
    const shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
    hex = hex.replace(shorthandRegex, (m, r, g, b) => {
      return r + r + g + g + b + b;
    });

    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? {
          r: parseInt(result[1], 16),
          g: parseInt(result[2], 16),
          b: parseInt(result[3], 16)
        }
      : null;
  };

  const interpolateColor = (start: RGBColor, end: RGBColor, factor: number) => {
    const result = {
      r: Math.round(start.r + (end.r - start.r) * factor),
      g: Math.round(start.g + (end.g - start.g) * factor),
      b: Math.round(start.b + (end.b - start.b) * factor)
    };
    return `rgba(${result.r}, ${result.g}, ${result.b}, 0.3)`;
  };

  const calculateGrid = (width: number, height: number) => {
    const columns = Math.ceil(width / charWidth);
    const rows = Math.ceil(height / charHeight);
    return { columns, rows };
  };

  const initializeLetters = (columns: number, rows: number) => {
    grid.current = { columns, rows };
    const totalLetters = columns * rows;
    letters.current = Array.from({ length: totalLetters }, () => ({
      char: getRandomChar(),
      color: getRandomColor(),
      targetColor: getRandomColor(),
      colorProgress: 1
    }));
  };

  const resizeCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = parent.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    if (context.current) {
      context.current.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    const { columns, rows } = calculateGrid(rect.width, rect.height);
    initializeLetters(columns, rows);
    detectAndHighlightArcane();

    drawLetters();
  };

  const drawLetters = () => {
    if (!context.current || letters.current.length === 0) return;
    const ctx = context.current;
    const { width, height } = canvasRef.current!.getBoundingClientRect();
    ctx.clearRect(0, 0, width, height);
    ctx.font = `bold ${fontSize}px monospace`;
    ctx.textBaseline = 'top';

    letters.current.forEach((letter, index) => {
      const x = (index % grid.current.columns) * charWidth;
      const y = Math.floor(index / grid.current.columns) * charHeight;
      
      if (letter.isHighlighted) {
        // Draw white background for ARCANE letters
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(x, y, charWidth, charHeight);
        
        // Draw dark text on white background
        ctx.fillStyle = '#000000'; // Dark font color
        ctx.fillText(letter.char, x, y);
      } else {
        // Normal drawing for non-highlighted letters
        ctx.fillStyle = letter.color;
        ctx.fillText(letter.char, x, y);
      }
    });
  };

  const updateLetters = () => {
    if (!letters.current || letters.current.length === 0) return;

    const updateCount = Math.max(1, Math.floor(letters.current.length * 0.05));

    for (let i = 0; i < updateCount; i++) {
      const index = Math.floor(Math.random() * letters.current.length);
      if (!letters.current[index]) continue;

      letters.current[index].char = getRandomChar();
      letters.current[index].targetColor = getRandomColor();

      if (!smooth) {
        letters.current[index].color = letters.current[index].targetColor;
        letters.current[index].colorProgress = 1;
      } else {
        letters.current[index].colorProgress = 0;
      }
    }
  };

  const handleSmoothTransitions = () => {
    let needsRedraw = false;
    letters.current.forEach(letter => {
      if (letter.colorProgress < 1) {
        letter.colorProgress += 0.05;
        if (letter.colorProgress > 1) letter.colorProgress = 1;

        const startRgb = hexToRgb(letter.color);
        const endRgb = hexToRgb(letter.targetColor);
        if (startRgb && endRgb) {
          letter.color = interpolateColor(startRgb, endRgb, letter.colorProgress);
          needsRedraw = true;
        }
      }
    });

    if (needsRedraw) {
      drawLetters();
    }
  };

  const animate = () => {
    const now = Date.now();
    if (now - lastGlitchTime.current >= glitchSpeed) {
      updateLetters();
      detectAndHighlightArcane();
      drawLetters();
      lastGlitchTime.current = now;
    }

    if (smooth) {
      handleSmoothTransitions();
    }

    animationRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    context.current = canvas.getContext('2d');
    resizeCanvas();
    animate();

    let resizeTimeout: NodeJS.Timeout;

    const handleResize = () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
        resizeCanvas();
        animate();
      }, 100);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      window.removeEventListener('resize', handleResize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [glitchSpeed, smooth]);

  const containerStyle = {
    position: 'relative',
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    overflow: 'hidden'
  } as const;

  const canvasStyle = {
    display: 'block',
    width: '100%',
    height: '100%'
  } as const;

  const outerVignetteStyle = {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
    background: 'radial-gradient(circle, rgba(185,223,224,0.05) 0%, rgba(242,148,192,0.02) 25%, rgba(199,133,242,0.02) 50%, rgba(0,0,0,0) 70%)'
  } as const;

  const centerVignetteStyle = {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
    background: 'radial-gradient(circle, rgba(131,91,217,0.1) 0%, rgba(157,228,250,0.05) 30%, rgba(0,0,0,0) 60%)'
  } as const;

  return (
    <div style={containerStyle} className={className}>
      <canvas ref={canvasRef} style={canvasStyle} />
      {outerVignette && <div style={outerVignetteStyle}></div>}
      {centerVignette && <div style={centerVignetteStyle}></div>}
    </div>
  );
};

export default LetterGlitch;