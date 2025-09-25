import React from 'react';
import { render, screen } from '@testing-library/react';
import { test, describe, expect } from 'vitest';
import { HighlightText } from '@/components/ui-custom/highlight-text';

describe('HighlightText', () => {
  test('renders text without highlighting when searchText is empty', () => {
    render(<HighlightText text="Hello world" searchText="" />);

    const text = screen.getByText('Hello world');
    expect(text).toBeInTheDocument();
    expect(text.querySelector('mark')).not.toBeInTheDocument();
  });

  test('renders text without highlighting when searchText is not provided', () => {
    render(<HighlightText text="Hello world" searchText={undefined as any} />);

    const text = screen.getByText('Hello world');
    expect(text).toBeInTheDocument();
    expect(text.querySelector('mark')).not.toBeInTheDocument();
  });

  test('highlights matching text case-insensitively', () => {
    const { container } = render(<HighlightText text="Hello World" searchText="world" />);

    const highlightedText = screen.getByText('World');
    expect(highlightedText.tagName).toBe('MARK');
    expect(container).toHaveTextContent('Hello World');
  });

  test('highlights multiple matches', () => {
    render(<HighlightText text="test this test" searchText="test" />);

    const marks = screen.getAllByText('test');
    expect(marks).toHaveLength(2);
    marks.forEach(mark => {
      expect(mark.tagName).toBe('MARK');
    });
  });

  test('escapes special regex characters in search text', () => {
    const { container } = render(<HighlightText text="Price: $10.99" searchText="$10.99" />);

    const highlightedText = screen.getByText('$10.99');
    expect(highlightedText.tagName).toBe('MARK');
    expect(container).toHaveTextContent('Price: $10.99');
  });

  test('applies custom className', () => {
    render(<HighlightText text="test" searchText="" className="custom-class" />);

    const container = screen.getByText('test');
    expect(container).toHaveClass('custom-class');
  });

  test('handles empty text', () => {
    render(<HighlightText text="" searchText="test" />);

    const container = screen.getByText('', { selector: 'span' });
    expect(container).toBeInTheDocument();
  });
});
