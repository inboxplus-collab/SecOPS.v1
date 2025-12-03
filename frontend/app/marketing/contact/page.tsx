export default function Contact() {
  return (
    <div className="bg-slate-950 text-white min-h-screen p-20">
      <h1 className="text-5xl font-extrabold">Talk with SecOpsAI</h1>
      <p className="text-lg text-slate-300 mt-4 max-w-2xl">
        Tell us about your environment and security objectives. We’ll assemble a tailored pilot that
        integrates with your cloud, CI/CD, and identity stack.
      </p>

      <div className="mt-8 max-w-xl">
        <form className="space-y-6">
          <div>
            <label className="block text-sm font-semibold">Work Email</label>
            <input
              type="email"
              className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-900 p-3 focus:border-blue-500 focus:outline-none"
              placeholder="you@company.com"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-semibold">Company</label>
            <input
              type="text"
              className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-900 p-3 focus:border-blue-500 focus:outline-none"
              placeholder="Your organization"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-semibold">What do you want to secure?</label>
            <textarea
              className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-900 p-3 focus:border-blue-500 focus:outline-none"
              rows={4}
              placeholder="Cloud footprint, CI/CD, Kubernetes, data stores, or all of the above"
            />
          </div>
          <button
            type="submit"
            className="w-full bg-blue-600 hover:bg-blue-700 py-3 rounded-lg font-semibold"
          >
            Request a demo
          </button>
        </form>
      </div>
    </div>
  );
}
