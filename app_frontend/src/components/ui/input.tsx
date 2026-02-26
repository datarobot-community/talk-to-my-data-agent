import * as React from 'react';

import { cn } from '@/lib/utils';

const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<'input'>>(
  ({ className, type, onKeyDown, ...props }, ref) => {
    const [isComposing, setIsComposing] = React.useState(false);
    return (
      <input
        type={type}
        onCompositionStart={() => setIsComposing(true)}
        onCompositionEnd={() => setIsComposing(false)}
        onKeyDown={event => {
          if (onKeyDown && !isComposing) {
            onKeyDown(event);
          }
        }}
        className={cn(
          'flex h-9 w-full rounded border border-border bg-input px-3 py-2 text-base shadow-xs transition-[color,box-shadow,border] duration-300 outline-none placeholder:text-muted-foreground hover:border-muted-foreground focus:border-accent disabled:cursor-not-allowed disabled:border-border/20 placeholder:disabled:text-muted-foreground/50 aria-invalid:border-destructive md:text-sm',
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = 'Input';

export { Input };
