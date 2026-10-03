import { createTheme } from '@mui/material/styles';

/**
 * Custom Material Design Theme for SwaraRegional
 * 
 * Specs:
 * - Primary Color: Deep Orange (#F57C00 / #FF5722)
 * - Clean, minimal Google Material 3 aesthetics
 * - Geometric pill shapes and rounded corners (border-radius: 12px to 24px)
 * - Modern typography using Inter / Roboto
 */
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#F57C00', // Deep Orange
      light: '#FFB74D',
      dark: '#E65100',
      contrastText: '#FFFFFF',
    },
    secondary: {
      main: '#263238', // Refined Blue-Grey / Slate
      light: '#455A64',
      dark: '#102027',
      contrastText: '#FFFFFF',
    },
    background: {
      default: '#F8F9FA', // Google-clean light gray/off-white
      paper: '#FFFFFF',
    },
    text: {
      primary: '#1F2937', // Refined high-contrast slate
      secondary: '#5F6368', // Material secondary gray
    },
    divider: 'rgba(0, 0, 0, 0.08)',
    action: {
      hover: 'rgba(245, 124, 0, 0.04)',
      selected: 'rgba(245, 124, 0, 0.08)',
    },
  },
  typography: {
    fontFamily: [
      'Roboto',
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'sans-serif',
    ].join(','),
    h4: {
      fontWeight: 700,
      letterSpacing: '-0.02em',
      color: '#1F2937',
    },
    h5: {
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h6: {
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    subtitle1: {
      fontWeight: 500,
      color: '#5F6368',
    },
    button: {
      textTransform: 'none', // Material Design 3 style
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 16, // Base geometric radius
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 24, // Pill shape
          padding: '8px 24px',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 4px 12px rgba(245, 124, 0, 0.25)',
          },
        },
        containedPrimary: {
          background: 'linear-gradient(135deg, #F57C00 0%, #FF5722 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #E65100 0%, #F4511E 100%)',
          },
        },
      },
    },
    MuiFab: {
      styleOverrides: {
        root: {
          borderRadius: 28,
          boxShadow: '0 6px 20px rgba(245, 124, 0, 0.35)',
          '&:hover': {
            boxShadow: '0 8px 26px rgba(245, 124, 0, 0.45)',
          },
        },
        primary: {
          backgroundColor: '#F57C00',
          '&:hover': {
            backgroundColor: '#E65100',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 18,
          border: '1px solid rgba(0, 0, 0, 0.06)',
          boxShadow: '0 2px 10px rgba(0, 0, 0, 0.03)',
          transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            boxShadow: '0 8px 24px rgba(245, 124, 0, 0.12)',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 16, // Geometric pill
          fontWeight: 500,
        },
        colorPrimary: {
          backgroundColor: 'rgba(245, 124, 0, 0.1)',
          color: '#E65100',
          '&:hover': {
            backgroundColor: 'rgba(245, 124, 0, 0.18)',
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#FFFFFF',
          color: '#1F2937',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
          borderBottom: '1px solid rgba(0, 0, 0, 0.06)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        rounded: {
          borderRadius: 18,
        },
      },
    },
  },
});

export default theme;
