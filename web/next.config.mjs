/** @type {import('next').NextConfig} */
const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "X-Frame-Options", value: "DENY" },
  // Microphone is needed by Live Trust on this origin only; everything else is off.
  { key: "Permissions-Policy", value: "microphone=(self), camera=(), geolocation=(), payment=()" },
];

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  eslint: {
    ignoreDuringBuilds: true,
  },
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
  // Hosted public checker: the Lab, Guardian and Control Center need the full backend, which it does not run.
  async redirects() {
    if (process.env.NEXT_PUBLIC_SITE_MODE !== "public") return [];
    return ["/", "/guardian", "/engine"].map((source) => ({ source, destination: "/check", permanent: false }));
  },
};

export default nextConfig;
