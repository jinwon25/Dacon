import { AR, KR, NL, PT, ZA } from 'country-flag-icons/react/3x2'

const flags = { ARG: AR, KOR: KR, NED: NL, POR: PT, RSA: ZA }

export default function CountryFlag({ code, label, className = '' }: { code: string; label: string; className?: string }) {
  const Flag = flags[code as keyof typeof flags]
  if (!Flag) return <span className={className} aria-label={label}>{code}</span>
  return <Flag className={className} aria-label={`${label} 국기`} />
}
