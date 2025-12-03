interface FeatureCardProps {
  title: string;
  desc: string;
}

export default function FeatureCard({ title, desc }: FeatureCardProps) {
  return (
    <div className="p-8 bg-gray-900 rounded-2xl shadow-lg border border-gray-800">
      <h3 className="text-2xl font-semibold mb-4">{title}</h3>
      <p className="text-gray-400 text-lg leading-relaxed">{desc}</p>
    </div>
  );
}
