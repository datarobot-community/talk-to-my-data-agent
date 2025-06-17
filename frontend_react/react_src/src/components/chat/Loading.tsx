export const Loading = () => {
  return (
    <div className="h-6 px-3 py-[7px] bg-muted rounded-[100px] justify-center items-center gap-1 flex">
      <div className="items-end gap-2.5 flex animate-bounce">
        <div className="w-2 h-2 bg-muted-foreground rounded-full" />
      </div>
      <div className="items-end gap-2.5 flex animate-bounce [animation-delay:-.3s]">
        <div className="w-2 h-2 bg-muted-foreground rounded-full" />
      </div>
      <div className="items-end gap-2.5 flex animate-bounce [animation-delay:-.5s]">
        <div className="w-2 h-2 bg-muted-foreground rounded-full" />
      </div>
    </div>
  );
};
