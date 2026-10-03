'use client';

import React, { useState, useEffect, useRef } from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Chip from '@mui/material/Chip';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import LinearProgress from '@mui/material/LinearProgress';
import Alert from '@mui/material/Alert';
import Paper from '@mui/material/Paper';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Tooltip from '@mui/material/Tooltip';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import TextField from '@mui/material/TextField';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import Divider from '@mui/material/Divider';

import RecordVoiceOverIcon from '@mui/icons-material/RecordVoiceOver';
import MicIcon from '@mui/icons-material/Mic';
import TextFieldsIcon from '@mui/icons-material/TextFields';
import GraphicEqIcon from '@mui/icons-material/GraphicEq';
import RefreshIcon from '@mui/icons-material/Refresh';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import CheckCircleOutlinedIcon from '@mui/icons-material/CheckCircleOutlined';
import PsychologyIcon from '@mui/icons-material/Psychology';
import StorageIcon from '@mui/icons-material/Storage';
import RssFeedIcon from '@mui/icons-material/RssFeed';
import SearchIcon from '@mui/icons-material/Search';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import ExploreOutlinedIcon from '@mui/icons-material/ExploreOutlined';

import VoiceRecorder from '../components/VoiceRecorder';
import RecommendationCard from '../components/RecommendationCard';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

const REGIONS = [
  { value: 'telangana', label: 'Telangana (Telugu / Hyderabadi)' },
  { value: 'tamil_nadu', label: 'Tamil Nadu (Chennai / Tamil)' },
  { value: 'karnataka', label: 'Karnataka (Bangalore / Kannada)' },
  { value: 'punjab', label: 'Punjab (Amritsar / Punjabi)' },
  { value: 'bengal', label: 'West Bengal (Kolkata / Bengali)' },
];

const CATEGORIES = [
  { value: '', label: 'All Categories (Auto-Detect via Zero-Shot)' },
  { value: 'food', label: 'Food & Cuisine' },
  { value: 'fashion', label: 'Handloom & Fashion' },
  { value: 'travel', label: 'Heritage & Travel' },
  { value: 'news', label: 'Regional News (Live RSS)' },
  { value: 'diaspora', label: 'Community & Diaspora' },
];

const PIPELINE_STEPS = [
  { label: 'Audio / Query Ingestion', desc: 'Capture 16kHz WAV or Text' },
  { label: 'Wav2Vec2 Accent', desc: 'Acoustic Phonetics on GPU' },
  { label: 'Whisper ASR', desc: 'Speech-to-Text Transcription' },
  { label: 'Zero-Shot Intent', desc: 'DistilBART NLI Classification' },
  { label: 'Intelligent Routing', desc: 'ChromaDB Store / Live RSS' },
];

const QUICK_PROMPTS = [
  {
    region: 'telangana',
    label: 'Telangana: Hyderabadi Biryani & Chai',
    icon: '🌶️',
    query: 'I want to eat authentic spicy Hyderabadi biryani and traditional snacks.',
  },
  {
    region: 'tamil_nadu',
    label: 'Tamil Nadu: Kanchipuram Mulberry Silk',
    icon: '🥻',
    query: 'I am looking for traditional Kanchipuram pure mulberry silk wedding sarees.',
  },
  {
    region: 'karnataka',
    label: 'Karnataka: Davangere Benne Dosa',
    icon: '☕',
    query: 'Where can I eat crisp Davangere benne dosa with potato palya?',
  },
  {
    region: 'punjab',
    label: 'Punjab: Golden Temple & Amritsari Kulcha',
    icon: '🌾',
    query: 'Tell me about visiting the Golden Temple and Harmandir Sahib in Amritsar.',
  },
  {
    region: 'bengal',
    label: 'Bengal: Kolkata Dum Biryani & Ilish',
    icon: '🐟',
    query: 'Where can I find tender Shorshe Ilish fish and authentic sweets in Kolkata?',
  },
  {
    region: 'telangana',
    label: 'Live News: Telangana Local Edition',
    icon: '📰',
    query: 'What are the latest news headlines and politics today in Hyderabad?',
  },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState(0); // 0 = Voice Studio, 1 = Text Studio
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Stepper Stage Tracker (0 to 5)
  const [activeStep, setActiveStep] = useState(0);
  const [processingStatusText, setProcessingStatusText] = useState('System ready. Record voice or enter a text query to trigger inference.');

  // Text Studio Form State
  const [textQuery, setTextQuery] = useState('');
  const [selectedRegion, setSelectedRegion] = useState('telangana');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [topK, setTopK] = useState(6);

  // Voice Studio Form State (Optional Query Hint & Audio File)
  const [voiceQueryHint, setVoiceQueryHint] = useState('');
  const [uploadedAudioFile, setUploadedAudioFile] = useState(null);

  // Results State
  const [recommendations, setRecommendations] = useState([]);
  const [accentData, setAccentData] = useState(null);
  const [transcriptionData, setTranscriptionData] = useState(null);
  const [intentData, setIntentData] = useState(null);
  const [routingDestination, setRoutingDestination] = useState(null);
  const [backendHealth, setBackendHealth] = useState(null);

  const fileInputRef = useRef(null);

  // Fetch backend health on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/v1/health`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setBackendHealth(data))
      .catch((err) => console.warn('FastAPI health check unavailable:', err));
  }, []);

  const simulateStepperProgress = () => {
    setActiveStep(0);
    setProcessingStatusText('Stage 1/5: Ingesting speech audio waveform tensor...');

    setTimeout(() => {
      setActiveStep(1);
      setProcessingStatusText('Stage 2/5: Extracting acoustic phonetic representations via Wav2Vec2 on CUDA...');
    }, 400);

    setTimeout(() => {
      setActiveStep(2);
      setProcessingStatusText('Stage 3/5: Transcribing acoustic phonemes to text via Whisper ASR...');
    }, 900);

    setTimeout(() => {
      setActiveStep(3);
      setProcessingStatusText('Stage 4/5: Classifying zero-shot domain intent with DistilBART-MNLI...');
    }, 1400);

    setTimeout(() => {
      setActiveStep(4);
      setProcessingStatusText('Stage 5/5: Querying ChromaDB vector index and dispatching live RSS feed requests...');
    }, 1900);
  };

  /**
   * Submit recorded audio blob to FastAPI backend
   */
  const handleAudioCaptured = async (audioBlob) => {
    setIsLoading(true);
    setError(null);
    simulateStepperProgress();

    const formData = new FormData();
    formData.append('file', audioBlob, 'recorded_speech.wav');
    formData.append('top_k', String(topK));
    if (voiceQueryHint.trim()) {
      formData.append('query_hint', voiceQueryHint.trim());
    }

    try {
      const response = await fetch(`${API_BASE}/api/v1/recommend-by-voice`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        throw new Error(errorJson.detail || `Server error (HTTP ${response.status})`);
      }

      const data = await response.json();
      setActiveStep(5);
      setProcessingStatusText('Pipeline execution complete! Recommendations successfully routed.');
      setAccentData(data.detected_accent);
      setTranscriptionData(data.transcription);
      setIntentData(data.intent_analysis);
      setRoutingDestination(data.routing_destination);
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error('Voice recommendation failed:', err);
      setError(err.message || 'Failed to process voice recommendation.');
      setActiveStep(0);
      setProcessingStatusText('Pipeline failed. Please verify audio input.');
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Submit uploaded audio file
   */
  const handleUploadedAudioSubmit = async () => {
    if (!uploadedAudioFile) return;

    setIsLoading(true);
    setError(null);
    simulateStepperProgress();

    const formData = new FormData();
    formData.append('file', uploadedAudioFile, uploadedAudioFile.name);
    formData.append('top_k', String(topK));
    if (voiceQueryHint.trim()) {
      formData.append('query_hint', voiceQueryHint.trim());
    }

    try {
      const response = await fetch(`${API_BASE}/api/v1/recommend-by-voice`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        throw new Error(errorJson.detail || `Server error (HTTP ${response.status})`);
      }

      const data = await response.json();
      setActiveStep(5);
      setProcessingStatusText('Pipeline execution complete! Recommendations successfully routed.');
      setAccentData(data.detected_accent);
      setTranscriptionData(data.transcription);
      setIntentData(data.intent_analysis);
      setRoutingDestination(data.routing_destination);
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error('Uploaded audio recommendation failed:', err);
      setError(err.message || 'Failed to process uploaded audio file.');
      setActiveStep(0);
      setProcessingStatusText('Pipeline failed. Please verify uploaded file.');
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Execute search from Text Studio
   */
  const handleTextSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!textQuery.trim()) {
      setError('Please enter a query in the text box.');
      return;
    }

    setIsLoading(true);
    setError(null);
    simulateStepperProgress();

    try {
      const payload = {
        query: textQuery.trim(),
        region: selectedRegion,
        category: selectedCategory || null,
        top_k: topK,
      };

      const response = await fetch(`${API_BASE}/api/v1/recommend-by-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        throw new Error(errorJson.detail || `Server error (HTTP ${response.status})`);
      }

      const data = await response.json();
      setActiveStep(5);
      setProcessingStatusText('Pipeline complete! Recommendations ready.');
      setAccentData(data.detected_accent);
      setTranscriptionData(data.transcription);
      setIntentData(data.intent_analysis);
      setRoutingDestination(data.routing_destination);
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error('Text search failed:', err);
      setError(err.message || 'Failed to retrieve recommendations.');
      setActiveStep(0);
      setProcessingStatusText('Pipeline failed. Please retry.');
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Execute 1-click regional prompt chip
   */
  const handleQuickPromptClick = async (prompt) => {
    setTextQuery(prompt.query);
    setSelectedRegion(prompt.region);
    setVoiceQueryHint(prompt.query);
    setIsLoading(true);
    setError(null);
    simulateStepperProgress();

    try {
      const response = await fetch(`${API_BASE}/api/v1/recommend-by-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: prompt.query,
          region: prompt.region,
          top_k: topK,
        }),
      });

      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        throw new Error(errorJson.detail || `Server error (HTTP ${response.status})`);
      }

      const data = await response.json();
      setActiveStep(5);
      setProcessingStatusText('Pipeline complete! Recommendations ready.');
      setAccentData(data.detected_accent);
      setTranscriptionData(data.transcription);
      setIntentData(data.intent_analysis);
      setRoutingDestination(data.routing_destination);
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error('Prompt recommendation failed:', err);
      setError(err.message || 'Failed to retrieve recommendations for prompt.');
      setActiveStep(0);
      setProcessingStatusText('Pipeline failed. Please retry.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetAll = () => {
    setRecommendations([]);
    setAccentData(null);
    setTranscriptionData(null);
    setIntentData(null);
    setRoutingDestination(null);
    setError(null);
    setActiveStep(0);
    setProcessingStatusText('System ready. Record voice or enter a text query to trigger inference.');
    setUploadedAudioFile(null);
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', pb: 10 }}>
      {/* 1. Clean Material App Bar Header */}
      <AppBar position="sticky" elevation={0}>
        <Toolbar sx={{ justifyContent: 'space-between', px: { xs: 2, md: 6 }, py: 0.5 }}>
          {/* Brand Logo & Title */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Box
              sx={{
                width: 42,
                height: 42,
                borderRadius: '14px',
                bgcolor: 'primary.main',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 4px 12px rgba(245, 124, 0, 0.3)',
              }}
            >
              <RecordVoiceOverIcon sx={{ color: '#FFFFFF', fontSize: 24 }} />
            </Box>
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 700, color: '#1F2937', lineHeight: 1.2 }}>
                SwaraRegional <Box component="span" sx={{ color: 'primary.main' }}>AI</Box>
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500 }}>
                Voice-Driven Regional Accent &amp; Hyperlocal Recommender
              </Typography>
            </Box>
          </Box>

          {/* Header Right Status & Docs Link */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            {backendHealth ? (
              <Chip
                icon={<CheckCircleOutlinedIcon sx={{ fontSize: 16, color: '#2E7D32 !important' }} />}
                label={`${backendHealth.torch_device?.toUpperCase()} Active • ${backendHealth.catalog_items_indexed} Items`}
                size="small"
                sx={{
                  bgcolor: 'rgba(46, 125, 50, 0.08)',
                  color: '#2E7D32',
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  display: { xs: 'none', sm: 'inline-flex' },
                }}
              />
            ) : (
              <Chip
                label="API Connecting..."
                size="small"
                sx={{ bgcolor: 'rgba(0, 0, 0, 0.06)', color: 'text.secondary', fontSize: '0.75rem' }}
              />
            )}

            <Tooltip title="View Interactive FastAPI Swagger Docs" arrow>
              <Button
                variant="outlined"
                size="small"
                startIcon={<MenuBookIcon sx={{ fontSize: 16 }} />}
                href={`${API_BASE}/docs`}
                target="_blank"
                rel="noopener noreferrer"
                sx={{
                  borderRadius: '20px',
                  fontWeight: 600,
                  fontSize: '0.8rem',
                  borderColor: 'rgba(0, 0, 0, 0.15)',
                  color: '#374151',
                  '&:hover': {
                    borderColor: 'primary.main',
                    bgcolor: 'rgba(245, 124, 0, 0.04)',
                  },
                }}
              >
                Docs
              </Button>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      {/* 2. Hero Section */}
      <Container maxWidth="lg" sx={{ pt: { xs: 3, md: 4 }, pb: 2, textAlign: 'center' }}>
        <Chip
          label="Google Material Design • Deep Orange Palette • Wav2Vec2 + Whisper + DistilBART"
          color="primary"
          sx={{ mb: 1.5, fontWeight: 600, fontSize: '0.78rem' }}
        />
        <Typography
          variant="h4"
          component="h1"
          sx={{
            fontWeight: 800,
            fontSize: { xs: '1.8rem', md: '2.5rem' },
            letterSpacing: '-0.02em',
            color: '#111827',
            mb: 1,
          }}
        >
          Indian Regional Voice &amp; Hyperlocal Recommender
        </Typography>
        <Typography
          variant="subtitle1"
          sx={{
            color: '#4B5563',
            maxWidth: 720,
            mx: 'auto',
            mb: 3,
            lineHeight: 1.5,
          }}
        >
          Speak in your natural accent or submit a query to identify regional dialects and discover authentic
          food, handloom textiles, heritage landmarks, and live state news.
        </Typography>

        {/* Global Error Banner */}
        {error && (
          <Alert severity="error" sx={{ mb: 3, borderRadius: '16px', textAlign: 'left' }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
      </Container>

      {/* 3. ALWAYS-VISIBLE Architecture & Processing Pipeline Stepper */}
      <Container maxWidth="lg" sx={{ mb: 3.5 }}>
        <Paper
          elevation={0}
          sx={{
            p: { xs: 2.5, md: 3 },
            borderRadius: '24px',
            bgcolor: '#FFFFFF',
            border: '1px solid rgba(245, 124, 0, 0.22)',
            boxShadow: '0 4px 20px rgba(245, 124, 0, 0.05)',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2.5, flexWrap: 'wrap', gap: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <AutoAwesomeIcon sx={{ color: 'primary.main', fontSize: 20 }} />
              <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#1F2937' }}>
                Multi-Stage AI Processing Pipeline
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {isLoading ? (
                <Chip
                  icon={<CircularProgress size={12} sx={{ color: 'primary.main' }} />}
                  label="Inference Running"
                  size="small"
                  sx={{
                    bgcolor: 'rgba(245, 124, 0, 0.1)',
                    color: 'primary.dark',
                    fontWeight: 700,
                    fontSize: '0.75rem',
                  }}
                />
              ) : activeStep === 5 ? (
                <Chip
                  icon={<CheckCircleOutlinedIcon sx={{ fontSize: 16, color: '#2E7D32 !important' }} />}
                  label="Pipeline Complete"
                  size="small"
                  sx={{
                    bgcolor: 'rgba(46, 125, 50, 0.1)',
                    color: '#2E7D32',
                    fontWeight: 700,
                    fontSize: '0.75rem',
                  }}
                />
              ) : (
                <Chip
                  label="Pipeline Ready"
                  size="small"
                  sx={{
                    bgcolor: 'rgba(0, 0, 0, 0.05)',
                    color: '#4B5563',
                    fontWeight: 600,
                    fontSize: '0.75rem',
                  }}
                />
              )}
            </Box>
          </Box>

          <Stepper
            activeStep={activeStep}
            alternativeLabel
            sx={{
              '& .MuiStepLabel-label.Mui-active': {
                color: 'primary.main',
                fontWeight: 700,
              },
              '& .MuiStepLabel-label.Mui-completed': {
                color: '#2E7D32',
                fontWeight: 600,
              },
              '& .MuiStepIcon-root.Mui-active': {
                color: 'primary.main',
              },
              '& .MuiStepIcon-root.Mui-completed': {
                color: '#2E7D32',
              },
            }}
          >
            {PIPELINE_STEPS.map((step, index) => (
              <Step key={step.label} completed={activeStep > index || activeStep === 5}>
                <StepLabel
                  optional={
                    <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', fontSize: '0.7rem' }}>
                      {step.desc}
                    </Typography>
                  }
                >
                  {step.label}
                </StepLabel>
              </Step>
            ))}
          </Stepper>

          {/* Dynamic Pipeline Progress Bar & Live Status Message */}
          <Box sx={{ mt: 2.5, pt: 1.5, borderTop: '1px solid rgba(0, 0, 0, 0.06)' }}>
            {isLoading && (
              <LinearProgress
                color="primary"
                sx={{
                  height: 6,
                  borderRadius: 3,
                  bgcolor: 'rgba(245, 124, 0, 0.12)',
                  mb: 1.2,
                }}
              />
            )}
            <Typography
              variant="body2"
              sx={{
                color: isLoading ? 'primary.dark' : '#374151',
                fontWeight: 600,
                fontSize: '0.85rem',
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <Box
                component="span"
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  bgcolor: isLoading ? 'primary.main' : activeStep === 5 ? '#2E7D32' : '#9CA3AF',
                  display: 'inline-block',
                }}
              />
              {processingStatusText}
            </Typography>
          </Box>
        </Paper>
      </Container>

      {/* 4. Main Interactive Studio: Dual Mode Tabs (Voice Studio vs Text Studio) */}
      <Container maxWidth="lg" sx={{ mb: 4 }}>
        <Paper
          elevation={0}
          sx={{
            p: { xs: 2.5, md: 3.5 },
            borderRadius: '24px',
            bgcolor: '#FFFFFF',
            border: '1px solid rgba(0, 0, 0, 0.08)',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.03)',
          }}
        >
          {/* Studio Tab Switcher */}
          <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
            <Paper
              elevation={0}
              sx={{
                p: 0.5,
                borderRadius: '24px',
                bgcolor: '#F3F4F6',
                display: 'inline-flex',
              }}
            >
              <Tabs
                value={activeTab}
                onChange={(e, val) => setActiveTab(val)}
                indicatorColor="primary"
                textColor="primary"
                sx={{
                  minHeight: 46,
                  '& .MuiTabs-indicator': {
                    height: '100%',
                    borderRadius: '20px',
                    bgcolor: 'primary.main',
                    zIndex: 0,
                  },
                }}
              >
                <Tab
                  icon={<MicIcon sx={{ fontSize: 20 }} />}
                  iconPosition="start"
                  label="Voice Studio (Microphone & Audio)"
                  sx={{
                    borderRadius: '20px',
                    minHeight: 46,
                    px: { xs: 2, sm: 3.5 },
                    fontWeight: 700,
                    textTransform: 'none',
                    fontSize: '0.9rem',
                    color: activeTab === 0 ? '#FFFFFF !important' : '#4B5563',
                    zIndex: 1,
                  }}
                />
                <Tab
                  icon={<TextFieldsIcon sx={{ fontSize: 20 }} />}
                  iconPosition="start"
                  label="Text Studio (Target Query & Filters)"
                  sx={{
                    borderRadius: '20px',
                    minHeight: 46,
                    px: { xs: 2, sm: 3.5 },
                    fontWeight: 700,
                    textTransform: 'none',
                    fontSize: '0.9rem',
                    color: activeTab === 1 ? '#FFFFFF !important' : '#4B5563',
                    zIndex: 1,
                  }}
                />
              </Tabs>
            </Paper>
          </Box>

          {/* TAB 0: VOICE STUDIO */}
          {activeTab === 0 && (
            <Box>
              <Grid container spacing={3} sx={{ alignItems: 'center' }}>
                {/* Left Column: Voice Recording FAB */}
                <Grid size={{ xs: 12, md: 5 }}>
                  <VoiceRecorder
                    onAudioCaptured={handleAudioCaptured}
                    isLoading={isLoading}
                    onReset={handleResetAll}
                  />
                </Grid>

                {/* Right Column: Audio File Upload & Guidance Hint */}
                <Grid size={{ xs: 12, md: 7 }}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 3,
                      borderRadius: '20px',
                      bgcolor: '#F9FAFB',
                      border: '1px solid rgba(0, 0, 0, 0.06)',
                    }}
                  >
                    <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#1F2937', mb: 1 }}>
                      Alternative Input: Upload Audio File
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'text.secondary', mb: 2 }}>
                      Upload pre-recorded WAV, MP3, OGG, or FLAC audio from regional speakers to test phonetic accent classification.
                    </Typography>

                    {/* File Dropzone Input */}
                    <Box
                      onClick={() => fileInputRef.current?.click()}
                      sx={{
                        p: 2.5,
                        borderRadius: '16px',
                        border: '2px dashed rgba(245, 124, 0, 0.35)',
                        bgcolor: '#FFFFFF',
                        cursor: 'pointer',
                        textAlign: 'center',
                        transition: 'all 0.2s ease',
                        '&:hover': {
                          borderColor: 'primary.main',
                          bgcolor: 'rgba(245, 124, 0, 0.03)',
                        },
                        mb: 2,
                      }}
                    >
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a"
                        style={{ display: 'none' }}
                        onChange={(e) => {
                          if (e.target.files && e.target.files[0]) {
                            setUploadedAudioFile(e.target.files[0]);
                          }
                        }}
                      />
                      <UploadFileIcon sx={{ color: 'primary.main', fontSize: 32, mb: 0.5 }} />
                      <Typography variant="body2" sx={{ fontWeight: 600, color: '#374151' }}>
                        {uploadedAudioFile ? uploadedAudioFile.name : 'Click to select or drop an audio file'}
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                        Supports 16kHz WAV, MP3, FLAC (Max 25MB)
                      </Typography>
                    </Box>

                    {/* Optional Query Hint for Voice */}
                    <TextField
                      fullWidth
                      size="small"
                      label="Optional Query Hint"
                      placeholder="e.g. I am looking for traditional Pochampally silk sarees"
                      value={voiceQueryHint}
                      onChange={(e) => setVoiceQueryHint(e.target.value)}
                      disabled={isLoading}
                      sx={{
                        mb: 2,
                        '& .MuiOutlinedInput-root': { borderRadius: '14px' },
                      }}
                    />

                    {/* Submit Uploaded Audio Button */}
                    {uploadedAudioFile && (
                      <Button
                        variant="contained"
                        color="primary"
                        fullWidth
                        onClick={handleUploadedAudioSubmit}
                        disabled={isLoading}
                        startIcon={isLoading ? <CircularProgress size={18} color="inherit" /> : <SearchIcon />}
                        sx={{ borderRadius: '20px', py: 1.2, fontWeight: 700 }}
                      >
                        {isLoading ? 'Processing Audio File...' : `Analyze ${uploadedAudioFile.name} & Recommend`}
                      </Button>
                    )}
                  </Paper>
                </Grid>
              </Grid>
            </Box>
          )}

          {/* TAB 1: TEXT STUDIO (Query & Metadata Targeting) */}
          {activeTab === 1 && (
            <Box component="form" onSubmit={handleTextSubmit}>
              <Box sx={{ mb: 2.5 }}>
                <Typography variant="h6" sx={{ fontWeight: 700, color: '#1F2937' }}>
                  Text Query &amp; Regional Intent Targeting
                </Typography>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  Directly search localized catalogs with zero-shot classification and semantic vector similarity.
                </Typography>
              </Box>

              <Grid container spacing={2.5}>
                {/* Search Query Input */}
                <Grid size={{ xs: 12 }}>
                  <TextField
                    fullWidth
                    multiline
                    rows={2}
                    placeholder="e.g. Where can I find authentic Chettinad pepper chicken and filter coffee in Chennai?"
                    value={textQuery}
                    onChange={(e) => setTextQuery(e.target.value)}
                    disabled={isLoading}
                    variant="outlined"
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        borderRadius: '16px',
                        fontSize: '1rem',
                      },
                    }}
                  />
                </Grid>

                {/* Target Region Dropdown */}
                <Grid size={{ xs: 12, sm: 4 }}>
                  <FormControl fullWidth size="medium">
                    <InputLabel id="region-select-label">Target Region</InputLabel>
                    <Select
                      labelId="region-select-label"
                      value={selectedRegion}
                      label="Target Region"
                      onChange={(e) => setSelectedRegion(e.target.value)}
                      disabled={isLoading}
                      sx={{ borderRadius: '14px' }}
                    >
                      {REGIONS.map((r) => (
                        <MenuItem key={r.value} value={r.value}>
                          {r.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                {/* Domain Category Dropdown */}
                <Grid size={{ xs: 12, sm: 4 }}>
                  <FormControl fullWidth size="medium">
                    <InputLabel id="category-select-label">Domain Intent</InputLabel>
                    <Select
                      labelId="category-select-label"
                      value={selectedCategory}
                      label="Domain Intent"
                      onChange={(e) => setSelectedCategory(e.target.value)}
                      disabled={isLoading}
                      sx={{ borderRadius: '14px' }}
                    >
                      {CATEGORIES.map((c) => (
                        <MenuItem key={c.value} value={c.value}>
                          {c.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                {/* Top-K Selector */}
                <Grid size={{ xs: 12, sm: 4 }}>
                  <FormControl fullWidth size="medium">
                    <InputLabel id="topk-select-label">Max Results (Top-K)</InputLabel>
                    <Select
                      labelId="topk-select-label"
                      value={topK}
                      label="Max Results (Top-K)"
                      onChange={(e) => setTopK(Number(e.target.value))}
                      disabled={isLoading}
                      sx={{ borderRadius: '14px' }}
                    >
                      <MenuItem value={3}>3 Recommendations</MenuItem>
                      <MenuItem value={6}>6 Recommendations</MenuItem>
                      <MenuItem value={9}>9 Recommendations</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                {/* Search Button */}
                <Grid size={{ xs: 12 }}>
                  <Button
                    type="submit"
                    variant="contained"
                    color="primary"
                    fullWidth
                    size="large"
                    disabled={isLoading}
                    startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
                    sx={{
                      borderRadius: '24px',
                      py: 1.5,
                      fontWeight: 700,
                      fontSize: '1.05rem',
                    }}
                  >
                    {isLoading ? 'Running Inference Pipeline...' : 'Search Localized Recommendations'}
                  </Button>
                </Grid>
              </Grid>
            </Box>
          )}

          {/* Quick Regional Presets (Available for both tabs) */}
          <Box sx={{ mt: 3, pt: 2.5, borderTop: '1px solid rgba(0, 0, 0, 0.06)' }}>
            <Typography
              variant="caption"
              sx={{
                display: 'block',
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                fontWeight: 700,
                color: 'text.secondary',
                mb: 1.5,
                textAlign: 'center',
              }}
            >
              1-Click Regional Test Presets
            </Typography>

            <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 1 }}>
              {QUICK_PROMPTS.map((prompt, idx) => (
                <Chip
                  key={idx}
                  label={`${prompt.icon} ${prompt.label}`}
                  onClick={() => handleQuickPromptClick(prompt)}
                  disabled={isLoading}
                  sx={{
                    borderRadius: '20px',
                    fontWeight: 600,
                    fontSize: '0.8rem',
                    bgcolor: '#FFFFFF',
                    border: '1px solid rgba(0, 0, 0, 0.1)',
                    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.02)',
                    transition: 'all 0.2s ease',
                    '&:hover': {
                      bgcolor: 'rgba(245, 124, 0, 0.08)',
                      borderColor: 'primary.light',
                      color: 'primary.dark',
                      transform: 'translateY(-1px)',
                    },
                  }}
                />
              ))}
            </Box>
          </Box>
        </Paper>
      </Container>

      {/* 5. ALWAYS-VISIBLE Inference Diagnostic Inspector */}
      <Container maxWidth="lg" sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <PsychologyIcon sx={{ color: 'primary.main', fontSize: 24 }} />
            <Typography variant="h6" sx={{ fontWeight: 700, color: '#1F2937' }}>
              Pipeline Inference Diagnostic Inspector
            </Typography>
          </Box>
          <Button
            size="small"
            startIcon={<RefreshIcon />}
            onClick={handleResetAll}
            sx={{ borderRadius: '16px', fontWeight: 600, color: '#6B7280' }}
          >
            Reset Inspector
          </Button>
        </Box>

        <Grid container spacing={2.5}>
          {/* Card 1: Regional Accent (Wav2Vec2) */}
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <Card elevation={0} sx={{ height: '100%', borderRadius: '20px', p: 2, border: '1px solid rgba(0, 0, 0, 0.07)' }}>
              <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700, textTransform: 'uppercase' }}>
                  1. Regional Accent (Wav2Vec2)
                </Typography>
                <Typography variant="h6" sx={{ fontWeight: 700, color: accentData ? 'primary.dark' : '#9CA3AF', mt: 0.5 }}>
                  {accentData ? accentData.region?.toUpperCase() : '—'}
                </Typography>
                <Typography variant="body2" sx={{ color: '#4B5563', fontSize: '0.85rem', mb: 1.5, minHeight: 22 }}>
                  {accentData ? accentData.detected_accent : 'Awaiting speech audio'}
                </Typography>

                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                    Confidence
                  </Typography>
                  <Typography variant="caption" sx={{ fontWeight: 700, color: 'primary.main' }}>
                    {accentData ? `${Math.round((accentData.confidence || 0.88) * 100)}%` : '0%'}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={accentData ? Math.round((accentData.confidence || 0.88) * 100) : 0}
                  sx={{ height: 6, borderRadius: 3, bgcolor: 'rgba(245, 124, 0, 0.12)' }}
                />

                <Box sx={{ mt: 1.5, display: 'flex', gap: 0.8 }}>
                  <Chip
                    label={accentData ? (accentData.device || 'CUDA') : (backendHealth?.torch_device?.toUpperCase() || 'CPU')}
                    size="small"
                    sx={{ fontSize: '0.7rem', fontWeight: 600 }}
                  />
                  {accentData?.checkpoint_loaded && (
                    <Chip label="Fine-Tuned" size="small" color="primary" sx={{ fontSize: '0.7rem', fontWeight: 600 }} />
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Card 2: Domain Intent (DistilBART NLI) */}
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <Card elevation={0} sx={{ height: '100%', borderRadius: '20px', p: 2, border: '1px solid rgba(0, 0, 0, 0.07)' }}>
              <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700, textTransform: 'uppercase' }}>
                  2. Domain Intent (DistilBART)
                </Typography>
                <Typography variant="h6" sx={{ fontWeight: 700, color: intentData ? '#1F2937' : '#9CA3AF', mt: 0.5 }}>
                  {intentData ? intentData.top_intent?.toUpperCase() : '—'}
                </Typography>
                <Typography variant="body2" sx={{ color: '#4B5563', fontSize: '0.85rem', mb: 1.5, minHeight: 22 }}>
                  {intentData ? 'Zero-Shot NLI Classification' : 'Awaiting semantic analysis'}
                </Typography>

                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                    Confidence
                  </Typography>
                  <Typography variant="caption" sx={{ fontWeight: 700, color: '#1F2937' }}>
                    {intentData ? `${Math.round((intentData.confidence || 0.9) * 100)}%` : '0%'}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={intentData ? Math.round((intentData.confidence || 0.9) * 100) : 0}
                  sx={{ height: 6, borderRadius: 3, bgcolor: 'rgba(0, 0, 0, 0.08)' }}
                />

                <Box sx={{ mt: 1.5 }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
                    Engine: {intentData?.engine || 'valhalla/distilbart-mnli'}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Card 3: Speech Transcription (Whisper) */}
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <Card elevation={0} sx={{ height: '100%', borderRadius: '20px', p: 2, border: '1px solid rgba(0, 0, 0, 0.07)' }}>
              <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700, textTransform: 'uppercase' }}>
                  3. Speech ASR (Whisper)
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    fontWeight: 500,
                    fontStyle: 'italic',
                    color: transcriptionData?.text ? '#1F2937' : '#9CA3AF',
                    mt: 0.8,
                    mb: 1.5,
                    minHeight: 44,
                  }}
                >
                  &ldquo;{transcriptionData?.text || 'Your spoken audio or query text will appear here...'}&rdquo;
                </Typography>

                <Box sx={{ mt: 'auto', display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip
                    label={transcriptionData?.status || 'idle'}
                    size="small"
                    sx={{
                      bgcolor: transcriptionData?.status === 'success' ? 'rgba(46, 125, 50, 0.08)' : 'rgba(0, 0, 0, 0.06)',
                      color: transcriptionData?.status === 'success' ? '#2E7D32' : '#6B7280',
                      fontWeight: 600,
                      fontSize: '0.7rem',
                    }}
                  />
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
                    Model: {transcriptionData?.model || 'openai/whisper'}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Card 4: Intelligent Routing Subsystem */}
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <Card elevation={0} sx={{ height: '100%', borderRadius: '20px', p: 2, border: '1px solid rgba(0, 0, 0, 0.07)' }}>
              <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700, textTransform: 'uppercase' }}>
                  4. Routing Subsystem
                </Typography>
                <Typography variant="h6" sx={{ fontWeight: 700, color: routingDestination ? '#1F2937' : '#9CA3AF', mt: 0.5 }}>
                  {routingDestination === 'rss_news_feed'
                    ? 'Live RSS News'
                    : routingDestination === 'chromadb_vector_store'
                    ? 'ChromaDB Store'
                    : 'Idle'}
                </Typography>
                <Typography variant="body2" sx={{ color: '#4B5563', fontSize: '0.85rem', mb: 1.5, minHeight: 22 }}>
                  {routingDestination === 'rss_news_feed'
                    ? 'The Hindu & Google News RSS'
                    : routingDestination === 'chromadb_vector_store'
                    ? 'Semantic vector similarity (all-MiniLM-L6)'
                    : 'Awaiting pipeline dispatch'}
                </Typography>

                <Chip
                  icon={
                    routingDestination === 'rss_news_feed' ? (
                      <RssFeedIcon sx={{ fontSize: 16 }} />
                    ) : (
                      <StorageIcon sx={{ fontSize: 16 }} />
                    )
                  }
                  label={`${recommendations.length} Localized Items`}
                  size="small"
                  sx={{
                    fontWeight: 600,
                    fontSize: '0.75rem',
                    bgcolor: recommendations.length > 0 ? 'rgba(245, 124, 0, 0.1)' : 'rgba(0, 0, 0, 0.06)',
                    color: recommendations.length > 0 ? 'primary.dark' : '#6B7280',
                  }}
                />
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>

      {/* 6. Localized Recommendations Grid Section */}
      <Container maxWidth="lg">
        <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1 }}>
          <Box>
            <Typography variant="h5" component="h2" sx={{ fontWeight: 700, color: '#1F2937' }}>
              Localized Recommendations
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
              Hyper-local recommendations tailored to your detected accent and cultural intent
            </Typography>
          </Box>
          {recommendations.length > 0 && (
            <Chip
              label={`Showing ${recommendations.length} item${recommendations.length === 1 ? '' : 's'}`}
              color="primary"
              variant="outlined"
              sx={{ fontWeight: 700 }}
            />
          )}
        </Box>

        {/* Empty State */}
        {recommendations.length === 0 && !isLoading && (
          <Paper
            elevation={0}
            sx={{
              p: { xs: 4, md: 6 },
              borderRadius: '24px',
              bgcolor: '#FFFFFF',
              border: '1px dashed rgba(0, 0, 0, 0.12)',
              textAlign: 'center',
            }}
          >
            <Box
              sx={{
                width: 64,
                height: 64,
                borderRadius: '50%',
                bgcolor: 'rgba(245, 124, 0, 0.08)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mx: 'auto',
                mb: 2,
              }}
            >
              <ExploreOutlinedIcon sx={{ fontSize: 36, color: 'primary.main' }} />
            </Box>
            <Typography variant="h6" sx={{ fontWeight: 700, color: '#1F2937', mb: 0.5 }}>
              Ready for Your Regional Voice or Query
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', maxWidth: 460, mx: 'auto', mb: 2.5 }}>
              Record your voice using the microphone FAB, upload an audio file, or click any of the 1-click test presets above.
            </Typography>
          </Paper>
        )}

        {/* Results Grid */}
        <Grid container spacing={3}>
          {recommendations.map((item, index) => (
            <Grid size={{ xs: 12, sm: 6, md: 4 }} key={item.id || index}>
              <RecommendationCard item={item} />
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
}
