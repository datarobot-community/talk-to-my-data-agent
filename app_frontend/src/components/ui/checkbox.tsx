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
        'peer group size-4 shrink-0 rounded-sm border border-primary text-primary-foreground ring-offset-background transition-all duration-200 ease-in hover:not-disabled:border-transparent focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:outline-none disabled:cursor-not-allowed disabled:border-muted-foreground disabled:opacity-70 data-[state=checked]:border-accent data-[state=checked]:bg-accent hover:data-[state=checked]:bg-accent/80 disabled:data-[state=checked]:border-muted-foreground disabled:data-[state=checked]:bg-muted-foreground data-[state=indeterminate]:border-accent data-[state=indeterminate]:bg-accent data-[state=indeterminate]:text-primary-foreground hover:data-[state=indeterminate]:bg-accent/80 disabled:data-[state=indeterminate]:border-muted-foreground disabled:data-[state=indeterminate]:bg-muted-foreground hover:not-disabled:data-[state=unchecked]:border-accent/80',
        className
      )}
      checked={checked}
      {...props}
    >
      <CheckboxPrimitive.Indicator className={cn('flex items-center justify-center')}>
        {isIndeterminate ? <Minus className="size-full" /> : <Check className="size-full" />}
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  );
});
Checkbox.displayName = CheckboxPrimitive.Root.displayName;

export { Checkbox };
