import DocsSidebar from "../components/DocsSidebar";

export default function Docs() {
  return (
    <div className="flex">
      <DocsSidebar />
      <main className="p-20 text-gray-200 max-w-3xl">
        <h1 className="text-6xl font-bold mb-6">Documentation</h1>
        <p className="text-xl">Learn how to install, configure, and integrate SecOpsAI.</p>

        <h2 id="installation" className="text-3xl mt-12 font-semibold">
          1. Installation
        </h2>
        <pre className="bg-gray-900 p-6 mt-4 rounded-lg">
          curl -sL https://secops.ai/install | bash
        </pre>

        <h2 id="api" className="text-3xl mt-12 font-semibold">
          2. API Reference
        </h2>
        <p className="mt-4 text-lg text-gray-300">
          All endpoints are authenticated via JWT. Use the <code>Authorization: Bearer &lt;token&gt;</code>
          header for every request. Rate limits and example payloads are documented in the platform
          console.
        </p>

        <h2 id="sso" className="text-3xl mt-12 font-semibold">
          3. SSO
        </h2>
        <p className="mt-4 text-lg text-gray-300">
          Enable Okta, Azure AD, or Google Workspace in the admin console. Redirect URIs should
          point to <code>/auth/&lt;provider&gt;/callback</code> in your deployment.
        </p>

        <h2 id="webhooks" className="text-3xl mt-12 font-semibold">
          4. Webhooks
        </h2>
        <p className="mt-4 text-lg text-gray-300">
          Receive real-time notifications for issue creation, remediation, and compliance drift.
          Configure webhook targets under Settings → Integrations.
        </p>
      </main>
    </div>
  );
}
