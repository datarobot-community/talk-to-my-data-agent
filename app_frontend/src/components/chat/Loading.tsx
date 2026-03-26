interface LoadingProps {
  statusText?: string;
}

export const Loading = ({ statusText }: LoadingProps) => {
  return (
    <div className="flex items-center gap-2">
      <div className="flex h-6 items-center justify-center gap-1 rounded-[100px] bg-[color-mix(in_srgb,var(--card)_95%,black)] px-3 py-[7px] dark:bg-[color-mix(in_srgb,var(--card)_85%,black)]">
        <div className="flex animate-bounce items-end gap-2.5">
          <div className="size-2 rounded-full bg-muted-foreground" />
        </div>
        <div className="flex animate-bounce items-end gap-2.5 [animation-delay:-.3s]">
          <div className="size-2 rounded-full bg-muted-foreground" />
        </div>
        <div className="flex animate-bounce items-end gap-2.5 [animation-delay:-.5s]">
          <div className="size-2 rounded-full bg-muted-foreground" />
        </div>
      </div>
      {statusText && <span className="body-secondary">{statusText}</span>}
    </div>
  );
};
