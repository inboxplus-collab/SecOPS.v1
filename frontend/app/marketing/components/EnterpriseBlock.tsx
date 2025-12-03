interface EnterpriseBlockProps {
  title: string;
  desc: string;
}

export default function EnterpriseBlock({ title, desc }: EnterpriseBlockProps) {
  return (
    <div className="p-6 bg-gray-900 rounded-xl border border-gray-800">
      <h3 className="text-2xl font-semibold mb-3">{title}</h3>
      <p className="text-gray-400 leading-relaxed">{desc}</p>
    </div>
  );
}
