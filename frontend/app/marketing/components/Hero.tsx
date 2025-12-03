export default function Hero() {
  return (
    <section className="px-20 py-40 bg-gradient-to-r from-blue-900 via-black to-black">
      <h1 className="text-7xl font-extrabold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent">
        Autonomous DevSecOps Intelligence
      </h1>
      <p className="mt-8 text-2xl max-w-2xl text-gray-300">
        The world’s first self-evolving security & reliability platform.
        Analyze → Detect → Explain → Auto-Fix.
      </p>
      <a
        href="/console"
        className="mt-12 inline-block bg-blue-600 hover:bg-blue-700 px-10 py-4 rounded-xl text-xl font-semibold"
      >
        Launch Console →
      </a>
    </section>
  );
}
