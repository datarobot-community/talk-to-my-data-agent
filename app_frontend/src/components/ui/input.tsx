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
          'h-9 rounded border border-border placeholder:text-muted-foreground aria-invalid:border-destructive bg-input flex field-sizing-content w-full px-3 py-2 text-base md:text-sm shadow-xs transition-[color,box-shadow,border] duration-300 outline-none hover:border-muted-foreground focus:border-accent disabled:border-border/20 placeholder:disabled:text-muted-foreground/50 disabled:cursor-not-allowed',
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
