import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="w-full border-t border-zinc-900 bg-black text-zinc-500 font-sans py-16 px-8 md:px-20">
      <div className="max-w-[1400px] mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 md:gap-0">
          {/* Brand Column */}
          <div className="md:border-r md:border-zinc-900 md:pr-12">
            <h2 className="text-4xl font-black text-white mb-8 tracking-tighter uppercase">A.R.C.A.N.E.</h2>
            <div className="space-y-4 text-[13px] font-mono leading-relaxed opacity-70">
              <Link 
                href="https://www.gpbacay.xyz/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="hover:text-white transition-colors cursor-pointer"
              >
                Gianne P. Bacay
              </Link>
              <p>Augmented Reconstruction of Consciousness through Artificial Neural Evolution</p>
              <p className="pt-6">
                <a href="mailto:giannebacay2004@gmail.com" className="hover:text-white transition-colors uppercase">giannebacay2004@gmail.com</a>
              </p>
            </div>
          </div>

          {/* Library Column */}
          <div className="md:border-r md:border-zinc-900 md:px-12">
            <h3 className="text-xs font-medium text-zinc-600 mb-6 lowercase">Library</h3>
            <ul className="space-y-4 text-[13px] font-bold text-zinc-100 tracking-wide uppercase">
              <li><Link href="/docs/layers" className="hover:text-[#C785F2] transition-colors">LAYERS</Link></li>
              <li><Link href="/docs/neuromimetic-model" className="hover:text-[#C785F2] transition-colors">MODELS</Link></li>
              <li><Link href="/docs/neural-resonance" className="hover:text-[#C785F2] transition-colors">MECHANISMS</Link></li>
              <li><Link href="/docs/resonant-gser" className="hover:text-[#C785F2] transition-colors">RESONANT GSER</Link></li>
              <li><Link href="/docs/cli" className="hover:text-[#C785F2] transition-colors">CLI TOOLS</Link></li>
              <li><Link href="/docs/benchmarks" className="hover:text-[#C785F2] transition-colors">BENCHMARKS</Link></li>
            </ul>
          </div>

          {/* Resources Column */}
          <div className="md:border-r md:border-zinc-900 md:px-12">
            <h3 className="text-xs font-medium text-zinc-600 mb-6 lowercase">Resources</h3>
            <ul className="space-y-4 text-[13px] font-bold text-zinc-100 tracking-wide uppercase">
              <li><Link href="/docs" className="hover:text-[#C785F2] transition-colors">DOCUMENTATION</Link></li>
              <li><Link href="/docs/quick-start" className="hover:text-[#C785F2] transition-colors">QUICK START</Link></li>
              <li><Link href="/docs/foundation-model" className="hover:text-[#C785F2] transition-colors">FOUNDATION MODEL</Link></li>
              <li><Link href="/docs/installation" className="hover:text-[#C785F2] transition-colors">INSTALLATION</Link></li>
              <li><Link href="https://github.com/gpbacay/gpbacay_arcane" className="hover:text-[#C785F2] transition-colors">GITHUB REPO</Link></li>
            </ul>
          </div>

          {/* Project Column */}
          <div className="md:px-12">
            <h3 className="text-xs font-medium text-zinc-600 mb-6 lowercase">More</h3>
            <ul className="space-y-4 text-[13px] font-bold text-zinc-100 tracking-wide uppercase">
              <li><Link href="/docs#research-applications" className="hover:text-[#C785F2] transition-colors">RESEARCH</Link></li>
              <li><Link href="https://github.com/gpbacay/gpbacay_arcane/blob/main/README.md#contributing" className="hover:text-[#C785F2] transition-colors">CONTRIBUTING</Link></li>
              <li><Link href="https://github.com/gpbacay/gpbacay_arcane/blob/main/LICENSE" className="hover:text-[#C785F2] transition-colors">LICENSE</Link></li>
              <li><Link href="https://github.com/gpbacay/gpbacay_arcane" className="hover:text-[#C785F2] transition-colors">SOURCE CODE</Link></li>
            </ul>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="mt-20 flex flex-col md:flex-row justify-between items-end gap-12 md:gap-0">
          <div className="max-w-sm">
            <p className="text-zinc-600 text-sm leading-relaxed font-medium">
              A Python library for building neuromimetic AI models inspired by 
              biological neural principles, bridging neuroscience and artificial intelligence.
            </p>
          </div>
          <div className="flex gap-16 text-[10px] font-medium tracking-[0.2em] text-zinc-600 uppercase">
            <Link 
              href="https://www.gpbacay.xyz/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="hover:text-white transition-colors cursor-pointer"
            >
              © {new Date().getFullYear()} GIANNE BACAY
            </Link>
            <p>MIT LICENSE</p>
          </div>
        </div>
      </div>
    </footer>
  );
}
