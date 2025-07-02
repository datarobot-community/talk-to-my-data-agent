# Talk to My Data: React App app_frontend

This application provides a modern React-based frontend for the **Talk to My Data** application. It allows users to interact with data, perform analyses, and chat with the system to gain insights from their datasets.

## Features

- Interactive chat interface for data analysis
- Data visualization with interactive plots
- Dataset management and cleansing
- Support for multiple data sources (CSV, Data Registry, Snowflake, Google Cloud)
- Code execution and insights generation

## Tech Stack

- React 18 with TypeScript
- Vite for fast development and building
- Tailwind CSS for styling
- Jest for testing
- React Query for API state management

## Development

See README in `app_backend` directory

## Testing

To run the test suite:

```bash
npm run test
```

## Project Structure

- `src/api`: API client and hooks for data fetching
- `src/components/ui`: shadcn components
- `src/components/ui-custom`: shadcn based generic components
- `src/pages`: Main application pages
- `src/state`: Application state management
- `src/assets`: Static assets like images and icons


## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default tseslint.config({
  extends: [
    // Remove ...tseslint.configs.recommended and replace with this
    ...tseslint.configs.recommendedTypeChecked,
    // Alternatively, use this for stricter rules
    ...tseslint.configs.strictTypeChecked,
    // Optionally, add this for stylistic rules
    ...tseslint.configs.stylisticTypeChecked,
  ],
  languageOptions: {
    // other options...
    parserOptions: {
      project: ['./tsconfig.node.json', './tsconfig.app.json'],
      tsconfigRootDir: import.meta.dirname,
    },
  },
})
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config({
  plugins: {
    // Add the react-x and react-dom plugins
    'react-x': reactX,
    'react-dom': reactDom,
  },
  rules: {
    // other rules...
    // Enable its recommended typescript rules
    ...reactX.configs['recommended-typescript'].rules,
    ...reactDom.configs.recommended.rules,
  },
})
```
