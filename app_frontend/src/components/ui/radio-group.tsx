'use client';

import * as React from 'react';
import * as RadioGroupPrimitive from '@radix-ui/react-radio-group';
import { CircleIcon } from 'lucide-react';

import { cn } from '@/lib/utils';

function RadioGroup({
  className,
  ...props
}: React.ComponentProps<typeof RadioGroupPrimitive.Root>) {
  return (
    <RadioGroupPrimitive.Root
      data-slot="radio-group"
      className={cn('grid gap-3', className)}
      {...props}
    />
  );
}

function RadioGroupItem({
  className,
  ...props
}: React.ComponentProps<typeof RadioGroupPrimitive.Item>) {
  return (
    <RadioGroupPrimitive.Item
      data-slot="radio-group-item"
      className={cn(
        // base
        'peer aspect-square size-4 shrink-0 rounded-full text-accent outline-none',
        // borders
        'border border-primary shadow-xs transition-[color,box-shadow]',
        // focus
        'focus-visible:border-ring focus-visible:ring-[1px] focus-visible:ring-ring',
        // invalid
        `
          aria-invalid:border-destructive aria-invalid:ring-destructive/20
          dark:aria-invalid:ring-destructive/40
        `,
        // disabled
        `
          disabled:cursor-not-allowed disabled:border-muted-foreground disabled:text-muted-foreground
          disabled:[&_[data-slot=radio-group-indicator]_svg]:fill-muted-foreground
        `,
        // checked
        'enabled:data-[state=checked]:border-accent',
        // hover
        `
          enabled:hover:border-[color-mix(in_srgb,var(--accent)_80%,white)] enabled:hover:text-[color-mix(in_srgb,var(--accent)_80%,white)]
          enabled:hover:data-[state=checked]:border-[color-mix(in_srgb,var(--accent)_80%,white)]
          enabled:hover:[&_[data-slot=radio-group-indicator]_svg]:fill-[color-mix(in_srgb,var(--accent)_80%,white)]
        `,
        className
      )}
      {...props}
    >
      <RadioGroupPrimitive.Indicator
        data-slot="radio-group-indicator"
        className="relative flex items-center justify-center"
      >
        <CircleIcon className="absolute top-1/2 left-1/2 size-2 -translate-1/2 fill-accent" />
      </RadioGroupPrimitive.Indicator>
    </RadioGroupPrimitive.Item>
  );
}

export { RadioGroup, RadioGroupItem };
