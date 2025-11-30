import { Card } from "@/components/shared/Card";
import { Input } from "@/components/shared/Input";
import { Button } from "@/components/shared/Button";

export default function SettingsPage() {
  return (
    <main className="flex min-h-screen flex-col bg-neutral-950 text-neutral-50">
      <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-4 px-4 py-6">
        <header>
          <h1 className="text-2xl font-semibold">Settings & Integrations</h1>
          <p className="text-sm text-neutral-400">
            Connect GitHub, Kubernetes clusters, and scanners to SecOpsAI.
          </p>
        </header>

        <Card className="space-y-4">
          <h2 className="text-sm font-medium text-neutral-200">GitHub</h2>
          <Input
            label="GitHub repository"
            placeholder="owner/repo"
            helperText="Used by CI hardening checks."
          />
          <Button variant="secondary">Save GitHub settings</Button>
        </Card>

        <Card className="space-y-4">
          <h2 className="text-sm font-medium text-neutral-200">Kubernetes</h2>
          <p className="text-xs text-neutral-500">
            Kubeconfig is read from the environment (in-cluster or ~/.kube/config).
          </p>
        </Card>
      </div>
    </main>
  );
}
