'use client';

import * as React from 'react';
import * as CheckboxPrimitive from '@radix-ui/react-checkbox';
import { Check, Minus } from 'lucide-react';

import { cn } from '@/lib/utils';

const Checkbox = React.forwardRef<
  React.ElementRef<typeof CheckboxPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof CheckboxPrimitive.Root>
>(({ className, checked, ...props }, ref) => {
  const isIndeterminate = checked === 'indeterminate';

  return (
    <CheckboxPrimitive.Root
      ref={ref}
      className={cn(
        'peer group h-4 w-4 shrink-0 hover:not-disabled:border-transparent hover:not-disabled:data-[state=unchecked]:border-accent/80 rounded-sm border border-primary ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-70 disabled:border-muted-foreground transition-all duration-200 ease-in data-[state=checked]:bg-accent  data-[state=checked]:border-accent hover:data-[state=checked]:bg-accent/80 disabled:data-[state=checked]:bg-muted-foreground disabled:data-[state=indeterminate]:bg-muted-foreground disabled:data-[state=checked]:border-muted-foreground disabled:data-[state=indeterminate]:border-muted-foreground text-primary-foreground data-[state=indeterminate]:bg-accent data-[state=indeterminate]:border-accent hover:data-[state=indeterminate]:bg-accent/80 data-[state=indeterminate]:text-primary-foreground',
        className
      )}
      checked={checked}
      {...props}
    >
      <CheckboxPrimitive.Indicator className={cn('flex items-center justify-center')}>
        {isIndeterminate ? (
          <Minus className="h-full w-full" />
        ) : (
          <Check className="h-full w-full" />
        )}
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  );
});
Checkbox.displayName = CheckboxPrimitive.Root.displayName;

export { Checkbox };
