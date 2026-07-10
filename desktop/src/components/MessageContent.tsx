import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { memo, useState } from 'react';
import { Check, Copy } from 'lucide-react';

// Renders assistant / user chat content as GitHub-flavoured markdown with
// syntax-highlighted code blocks and a per-block copy button.
export const MessageContent = memo(function MessageContent({ content }: { content: string }) {
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
          }
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

  return (
    <div className="code-block-wrapper group/code">
      {language && <span className="code-block-lang">{language}</span>}
      <button
        className="code-block-copy flex items-center gap-1"
        onClick={handleCopy}
        title="Copy code"
        aria-label="Copy code"
      >
        {copied ? <Check size={13} /> : <Copy size={13} />}
        <span className="text-[11px]">{copied ? 'Copied' : 'Copy'}</span>
      </button>
      <code className={className} {...props}>{children}</code>
    </div>
  );
}

export default MessageContent;
