import Link from "next/link";
import PrismaticBurst from "@/components/PrismaticBurst";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-black text-white selection:bg-[#C785F2]/30 overflow-hidden relative font-sans">
      
      {/* PrismaticBurst Background */}
      <PrismaticBurst
        animationType="rotate3d"
        intensity={2.8}
        speed={0.4}
        distort={1.2}
        paused={false}
        offset={{ x: 0, y: 0 }}
        hoverDampness={0.25}
        rayCount={32}
        mixBlendMode="lighten"
        colors={['#B9DFE0', '#F294C0', '#C785F2', '#835BD9', '#9DE4FA', '#8B5CF6']}
      />

      <main className="z-10 flex flex-col items-center text-center px-4 animate-in fade-in zoom-in duration-1000 max-w-5xl">
        <h1 className="text-7xl font-extrabold tracking-tighter sm:text-9xl bg-clip-text text-transparent bg-gradient-to-b from-white via-white to-white/40 mb-6 drop-shadow-2xl">
          A.R.C.A.N.E.
        </h1>
        
        <p className="text-lg sm:text-xl font-medium text-[#B9DFE0] mb-8 tracking-[0.15em] uppercase px-4 max-w-3xl opacity-90 leading-relaxed">
          Augmented Reconstruction of Consciousness through Artificial Neural Evolution
        </p>
        
        <p className="max-w-3xl text-base sm:text-lg md:text-xl text-zinc-300 mb-12 leading-relaxed opacity-80 backdrop-blur-sm bg-black/10 py-4 px-4 sm:px-6 rounded-2xl">
          A Python library for building neuromimetic AI models inspired by biological neural principles.
          Bridging the gap between neuroscience and artificial intelligence through
          biologically-plausible neural architectures.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 sm:gap-6 w-full sm:w-auto">
          <Link
            href="/docs"
            className="group relative inline-flex h-12 sm:h-14 items-center justify-center overflow-hidden rounded-full bg-white px-6 sm:px-10 font-bold text-black transition-all hover:bg-zinc-200 hover:scale-105 active:scale-95 shadow-[0_0_30px_rgba(255,255,255,0.2)] text-sm sm:text-base"
          >
            <span className="mr-2 sm:mr-3">Explore Docs</span>
            <svg className="h-4 w-4 sm:h-5 sm:w-5 transition-transform group-hover:translate-x-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </Link>
          <a
            href="https://github.com/gpbacay/gpbacay_arcane"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex h-12 sm:h-14 items-center justify-center rounded-full border border-zinc-700 bg-black/40 backdrop-blur-md px-6 sm:px-10 font-bold text-white transition-all hover:bg-zinc-900/60 hover:border-[#C785F2] hover:scale-105 active:scale-95 text-sm sm:text-base"
          >
            GitHub Repo
          </a>
        </div>
      </main>
    </div>
  );
}
