interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = "" }: SkeletonProps) {
  return (
    <div
      className={`animate-pulse rounded bg-bg-tertiary ${className}`}
      style={{
        animation: 'skeleton-pulse 1.5s ease-in-out infinite',
      }}
    />
  );
}

export function SkeletonText({ lines = 3, className = "" }: { lines?: number; className?: string }) {
  return (
    <div className={className}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="mb-2 rounded bg-bg-tertiary animate-pulse"
          style={{
            height: '16px',
            width: i === 0 ? '100%' : i === lines - 1 ? '60%' : '85%',
            animation: 'skeleton-pulse 1.5s ease-in-out infinite',
          }}
        />
      ))}
    </div>
  );
}

export function SkeletonCard({ className = "" }: { className?: string }) {
  return (
    <div className={`rounded-lg border border-border bg-bg-secondary p-4 ${className}`}>
      <div className="flex items-center gap-3 mb-3">
        <div className="w-8 h-8 rounded-full bg-bg-tertiary animate-pulse" />
        <div className="flex-1 space-y-1">
          <div className="h-4 w-32 rounded bg-bg-tertiary animate-pulse" />
          <div className="h-3 w-24 rounded bg-bg-tertiary animate-pulse" />
        </div>
      </div>
      <SkeletonText lines={3} />
    </div>
  );
}