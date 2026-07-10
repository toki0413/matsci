/** Skeleton placeholder — shimmer animation, matches typical panel layout. */

export function SkeletonRow({ lines = 3 }: { lines?: number }) {
  return (
    <div className="space-y-2 p-3">
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-3.5 rounded bg-bg-tertiary"
          style={{
            width: `${[100, 85, 70, 92][i % 4]}%`,
            animation: `skeleton-pulse 1.5s ease-in-out ${i * 0.1}s infinite`,
          }}
        />
      ))}
    </div>
  );
}

export function SkeletonList({ items = 4 }: { items?: number }) {
  return (
    <div className="space-y-3 p-3">
      {Array.from({ length: items }).map((_, i) => (
        <div key={i} className="flex items-center gap-3">
          <div
            className="h-8 w-8 rounded-full bg-bg-tertiary"
            style={{ animation: `skeleton-pulse 1.5s ease-in-out ${i * 0.1}s infinite` }}
          />
          <div className="flex-1 space-y-1.5">
            <div
              className="h-3 rounded bg-bg-tertiary"
              style={{ width: "60%", animation: `skeleton-pulse 1.5s ease-in-out ${i * 0.1}s infinite` }}
            />
            <div
              className="h-2.5 rounded bg-bg-tertiary"
              style={{ width: "40%", animation: `skeleton-pulse 1.5s ease-in-out ${i * 0.15}s infinite` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export function SkeletonCard() {
  return (
    <div className="card p-4 space-y-3">
      <div
        className="h-4 w-1/3 rounded bg-bg-tertiary"
        style={{ animation: "skeleton-pulse 1.5s ease-in-out infinite" }}
      />
      <div
        className="h-3 w-full rounded bg-bg-tertiary"
        style={{ animation: "skeleton-pulse 1.5s ease-in-out 0.1s infinite" }}
      />
      <div
        className="h-3 w-4/5 rounded bg-bg-tertiary"
        style={{ animation: "skeleton-pulse 1.5s ease-in-out 0.2s infinite" }}
      />
    </div>
  );
}
