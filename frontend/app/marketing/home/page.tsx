export default function Home() {
  return (
    <div className="bg-black text-white min-h-screen p-20">
      <h1 className="text-6xl font-extrabold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent">
        SecOpsAI
      </h1>

      <p className="text-xl mt-6 max-w-2xl">
        The world’s first Autonomous DevSecOps Intelligence Engine.
        Instantly analyze infrastructure, detect vulnerabilities, and apply AI-powered fixes.
      </p>

      <a
        href="/console"
        className="mt-10 inline-block bg-blue-600 hover:bg-blue-700 px-8 py-4 rounded-xl text-lg font-semibold"
      >
        Launch Console →
      </a>
    </div>
  );
}
