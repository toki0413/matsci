import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { memo } from 'react';

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
            return (
              <div className="code-block-wrapper">
                {match && <span className="code-block-lang">{match[1]}</span>}
                <button
                  className="code-block-copy"
                  onClick={(e) => {
                    const btn = e.currentTarget;
                    const text = String(children).replace(/\n$/, '');
                    navigator.clipboard.writeText(text)
                      .then(() => {
                        btn.textContent = '✓';
                        setTimeout(() => { btn.textContent = '📋'; }, 1500);
                      })
                      .catch(() => {
                        // Fallback for browsers without clipboard API
                        const ta = document.createElement('textarea');
                        ta.value = text;
                        document.body.appendChild(ta);
                        ta.select();
                        document.execCommand('copy');
                        document.body.removeChild(ta);
                        btn.textContent = '✓';
                        setTimeout(() => { btn.textContent = '📋'; }, 1500);
                      });
                  }}
                  title="Copy"
                >📋</button>
                <code className={className} {...props}>{children}</code>
              </div>
            );
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

export default MessageContent;
