const links = [
  { href: "/marketing/docs", label: "Getting Started" },
  { href: "/marketing/docs#installation", label: "Installation" },
  { href: "/marketing/docs#api", label: "API Reference" },
  { href: "/marketing/docs#sso", label: "SSO" },
  { href: "/marketing/docs#webhooks", label: "Webhooks" },
];

export default function DocsSidebar() {
  return (
    <aside className="w-64 bg-gray-950 border-r border-gray-800 min-h-screen p-8 text-gray-300">
      <h3 className="text-xl font-semibold mb-6">Developer Docs</h3>
      <nav className="space-y-4">
        {links.map((item) => (
          <a key={item.href} href={item.href} className="block hover:text-white">
            {item.label}
          </a>
        ))}
      </nav>
    </aside>
  );
}
