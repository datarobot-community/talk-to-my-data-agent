interface LoadingProps {
  statusText?: string;
}

export const Loading = ({ statusText }: LoadingProps) => {
  return (
    <div className="flex items-center gap-2">
      <div className="h-6 px-3 py-[7px] bg-[color-mix(in_srgb,var(--card)_95%,black)] dark:bg-[color-mix(in_srgb,var(--card)_85%,black)] rounded-[100px] justify-center items-center gap-1 flex">
        <div className="items-end gap-2.5 flex animate-bounce">
          <div className="w-2 h-2 rounded-full bg-muted-foreground" />
        </div>
        <div className="items-end gap-2.5 flex animate-bounce [animation-delay:-.3s]">
          <div className="w-2 h-2 rounded-full bg-muted-foreground" />
        </div>
        <div className="items-end gap-2.5 flex animate-bounce [animation-delay:-.5s]">
          <div className="w-2 h-2 rounded-full bg-muted-foreground" />
        </div>
      </div>
      {statusText && <span className="body-secondary">{statusText}</span>}
    </div>
  );
};
