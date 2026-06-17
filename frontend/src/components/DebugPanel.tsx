import { useState } from 'react'

export interface DebugInfo {
  timestamp: string
  status?: number
  url?: string
  method?: string
  request?: Record<string, unknown>
  responseBody?: unknown
  responseHeaders?: Record<string, string>
  errorMessage: string
  errorStack?: string
}

interface DebugPanelProps {
  info: DebugInfo
}

/** Error panel dengan debug detail + copy-to-clipboard untuk easy bug report. */
export default function DebugPanel({ info }: DebugPanelProps) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)

  const formattedDebug = formatDebugInfo(info)

  async function copyToClipboard() {
    try {
      await navigator.clipboard.writeText(formattedDebug)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Copy failed:', err)
    }
  }

  return (
    <div className="error" role="alert">
      <div className="error-header">
        <strong>{info.errorMessage}</strong>
        <button
          type="button"
          className="error-toggle"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? 'Sembunyikan debug ▲' : 'Tampilkan debug ▼'}
        </button>
      </div>

      {expanded && (
        <div className="error-debug">
          <div className="debug-actions">
            <button
              type="button"
              className="debug-copy"
              onClick={copyToClipboard}
            >
              {copied ? '✓ Tersalin' : 'Copy debug info'}
            </button>
            <a
              href="/api/status"
              target="_blank"
              rel="noreferrer"
              className="debug-link"
            >
              Open /api/status →
            </a>
          </div>
          <pre className="debug-pre">{formattedDebug}</pre>
        </div>
      )}
    </div>
  )
}

function formatDebugInfo(info: DebugInfo): string {
  const lines: string[] = []
  lines.push('=== KozyNear Debug Info ===')
  lines.push(`Timestamp:     ${info.timestamp}`)
  lines.push(`Error message: ${info.errorMessage}`)
  if (info.status !== undefined) {
    lines.push(`HTTP status:   ${info.status}`)
  }
  if (info.method && info.url) {
    lines.push(`Request:       ${info.method} ${info.url}`)
  }
  if (info.request) {
    lines.push('Request params:')
    lines.push(indent(JSON.stringify(info.request, null, 2)))
  }
  if (info.responseHeaders && Object.keys(info.responseHeaders).length > 0) {
    lines.push('Response headers:')
    for (const [k, v] of Object.entries(info.responseHeaders)) {
      lines.push(`  ${k}: ${v}`)
    }
  }
  if (info.responseBody !== undefined && info.responseBody !== null) {
    lines.push('Response body:')
    const body =
      typeof info.responseBody === 'string'
        ? info.responseBody
        : JSON.stringify(info.responseBody, null, 2)
    lines.push(indent(body))
  }
  if (info.errorStack) {
    lines.push('Stack trace:')
    lines.push(indent(info.errorStack))
  }
  lines.push('=============================')
  return lines.join('\n')
}

function indent(text: string): string {
  return text
    .split('\n')
    .map((line) => '  ' + line)
    .join('\n')
}
