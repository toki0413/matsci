import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { memo, useState, useMemo, useCallback } from 'react';
import { Check, Copy, ChevronDown, ChevronUp, ZoomIn } from 'lucide-react';

// Renders assistant / user chat content as GitHub-flavoured markdown with
// syntax-highlighted code blocks and a per-block copy button.
export const MessageContent = memo(function MessageContent({ content }: { content: string }) {
  const [zoomedImg, setZoomedImg] = useState<string | null>(null);

  const imgComponent = useCallback(({ src, alt, ...props }: any) => (
    <span className="inline-block relative group/img my-2">
      <img
        src={src}
        alt={alt}
        className="max-w-full rounded-lg border border-border cursor-zoom-in"
        onClick={() => setZoomedImg(src)}
        loading="lazy"
        {...props}
      />
      <span className="absolute bottom-1 right-1 rounded bg-bg-secondary/80 p-1 opacity-0 group-hover/img:opacity-100 transition-opacity pointer-events-none">
        <ZoomIn size={12} className="text-text-muted" />
      </span>
      {zoomedImg === src && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-8 cursor-zoom-out"
          onClick={() => setZoomedImg(null)}
        >
          <img src={src} alt={alt} className="max-w-full max-h-full rounded-lg shadow-2xl" />
        </div>
      )}
    </span>
  ), [zoomedImg]);

  return (
    <div className="chat-prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            const isInline = !match && !className;
            if (isInline) {
              return <code {...props}>{children}</code>;
            }
            return <CodeBlock language={match?.[1]} code={String(children).replace(/\n$/, '')} className={className} props={props}>{children}</CodeBlock>;
          },
          pre({ children }) {
            return <pre>{children}</pre>;
          },
          img: imgComponent,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});

function CodeBlock({ language, code, className, props, children }: {
  language?: string;
  code: string;
  className?: string;
  props: Record<string, unknown>;
  children: React.ReactNode;
}) {
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(true);
  const lineCount = useMemo(() => code.split('\n').length, [code]);
  const isLong = lineCount > 30;

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
      .then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); })
      .catch(() => {
        const ta = document.createElement('textarea');
        ta.value = code;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      });
  };

  const lineNumbers = useMemo(() => {
    const lines = expanded || !isLong ? code.split('\n') : code.split('\n').slice(0, 20);
    return lines.map((_, i) => i + 1).join('\n');
  }, [code, expanded, isLong]);

  return (
    <div className="code-block-wrapper group/code">
      <div className="flex items-center justify-between px-3 py-1 border-b border-border/50 bg-bg-tertiary/50">
        {language && <span className="code-block-lang !static">{language}</span>}
        <div className="flex items-center gap-2 ml-auto">
          {isLong && (
            <button
              className="flex items-center gap-1 text-[11px] text-text-muted hover:text-text-primary"
              onClick={() => setExpanded(!expanded)}
              title={expanded ? 'Collapse' : 'Expand'}
            >
              {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              {expanded ? 'Collapse' : `Expand (${lineCount} lines)`}
            </button>
          )}
          <button
            className="flex items-center gap-1 text-[11px] text-text-muted hover:text-text-primary"
            onClick={handleCopy}
            title="Copy code"
            aria-label="Copy code"
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>
      <div className="flex">
        <pre
          className="select-none text-right text-text-muted/40 leading-[1.5] py-3 pl-3 pr-2 text-[12.5px] font-mono"
          aria-hidden="true"
          style={{ margin: 0, background: 'none', border: 'none' }}
        >
          {lineNumbers}
        </pre>
        <code className={className} {...props} style={{ display: 'block', overflow: 'hidden' }}>
          {expanded || !isLong ? children : (code.split('\n').slice(0, 20).join('\n') + '\n…')}
        </code>
      </div>
    </div>
  );
}

export default MessageContent;
