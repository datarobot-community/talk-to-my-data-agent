'use client';

import * as React from 'react';
import * as ToggleGroupPrimitive from '@radix-ui/react-toggle-group';

import { cn } from '@/lib/utils';

function ToggleGroup({
  className,
  ...props
}: React.ComponentProps<typeof ToggleGroupPrimitive.Root>) {
  return (
    <ToggleGroupPrimitive.Root
      data-slot="toggle-group"
      className={cn(
        'inline-flex items-center rounded-lg border-2 border-sidebar-border bg-sidebar-border',
        className
      )}
      {...props}
    />
  );
}

function ToggleGroupItem({
  className,
  ...props
}: React.ComponentProps<typeof ToggleGroupPrimitive.Item>) {
  return (
    <ToggleGroupPrimitive.Item
      data-slot="toggle-group-item"
      className={cn(
        // Base styles
        'inline-flex items-center justify-center',
        // Spacing
        'px-4 py-1.5',
        // Typography
        'text-sm font-medium',
        // Background & Text
        'text-secondary-foreground',
        // Shape & Border
        'rounded-lg',
        // Cursor & Interactivity
        `
          cursor-pointer
          disabled:cursor-default
        `,
        // Focus state
        `
          outline-hidden
          focus:bg-muted focus:text-accent
          focus-visible:border-ring focus-visible:ring-[1px] focus-visible:ring-ring
        `,
        // Hover state
        'hover:bg-muted hover:text-accent',
        // Active/Pressed state
        `
          focus:z-2
          data-[state=on]:border-accent data-[state=on]:bg-sidebar-accent data-[state=on]:text-foreground
        `,
        // Disabled state
        'data-[disabled]:pointer-events-none data-[disabled]:text-muted-foreground data-[disabled]:opacity-50',
        // Text selection
        'select-none',
        // Transitions
        'transition-colors',
        className
      )}
      {...props}
    />
  );
}

export { ToggleGroup, ToggleGroupItem };
