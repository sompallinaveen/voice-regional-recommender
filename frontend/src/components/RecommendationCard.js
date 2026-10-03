'use client';

import React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardActions from '@mui/material/CardActions';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Button from '@mui/material/Button';
import LocationOnOutlinedIcon from '@mui/icons-material/LocationOnOutlined';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import RestaurantMenuIcon from '@mui/icons-material/RestaurantMenu';
import CheckroomIcon from '@mui/icons-material/Checkroom';
import ExploreIcon from '@mui/icons-material/Explore';
import NewspaperIcon from '@mui/icons-material/Newspaper';
import GroupsIcon from '@mui/icons-material/Groups';
import PlaceIcon from '@mui/icons-material/Place';

// Minimalist, geometric domain themes (strictly non-neon)
const CATEGORY_CONFIG = {
  food: {
    label: 'Food & Cuisine',
    icon: <RestaurantMenuIcon sx={{ fontSize: 16 }} />,
    placeholderIcon: <RestaurantMenuIcon sx={{ fontSize: 44, color: '#F57C00', opacity: 0.85 }} />,
    color: '#E65100', // Deep Orange
    bgColor: 'rgba(245, 124, 0, 0.08)',
    gradient: 'linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%)',
  },
  fashion: {
    label: 'Handloom & Fashion',
    icon: <CheckroomIcon sx={{ fontSize: 16 }} />,
    placeholderIcon: <CheckroomIcon sx={{ fontSize: 44, color: '#D84315', opacity: 0.85 }} />,
    color: '#D84315',
    bgColor: 'rgba(216, 67, 21, 0.08)',
    gradient: 'linear-gradient(135deg, #FBE9E7 0%, #FFCCBC 100%)',
  },
  travel: {
    label: 'Heritage & Travel',
    icon: <ExploreIcon sx={{ fontSize: 16 }} />,
    placeholderIcon: <ExploreIcon sx={{ fontSize: 44, color: '#EF6C00', opacity: 0.85 }} />,
    color: '#EF6C00',
    bgColor: 'rgba(239, 108, 0, 0.08)',
    gradient: 'linear-gradient(135deg, #FFF8E1 0%, #FFE082 100%)',
  },
  news: {
    label: 'Regional News',
    icon: <NewspaperIcon sx={{ fontSize: 16 }} />,
    placeholderIcon: <NewspaperIcon sx={{ fontSize: 44, color: '#37474F', opacity: 0.85 }} />,
    color: '#37474F', // Refined Slate
    bgColor: 'rgba(55, 71, 79, 0.08)',
    gradient: 'linear-gradient(135deg, #ECEFF1 0%, #CFD8DC 100%)',
  },
  diaspora: {
    label: 'Community & Diaspora',
    icon: <GroupsIcon sx={{ fontSize: 16 }} />,
    placeholderIcon: <GroupsIcon sx={{ fontSize: 44, color: '#4E342E', opacity: 0.85 }} />,
    color: '#4E342E', // Warm Charcoal
    bgColor: 'rgba(78, 52, 46, 0.08)',
    gradient: 'linear-gradient(135deg, #EFEBE9 0%, #D7CCC8 100%)',
  },
};

const REGION_NAMES = {
  telangana: 'Telangana',
  tamil_nadu: 'Tamil Nadu',
  karnataka: 'Karnataka',
  punjab: 'Punjab',
  bengal: 'West Bengal',
};

export default function RecommendationCard({ item }) {
  if (!item) return null;

  const categoryKey = (item.category || 'general').toLowerCase();
  const config = CATEGORY_CONFIG[categoryKey] || {
    label: item.category || 'General',
    icon: null,
    placeholderIcon: <PlaceIcon sx={{ fontSize: 44, color: '#F57C00' }} />,
    color: '#F57C00',
    bgColor: 'rgba(245, 124, 0, 0.08)',
    gradient: 'linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%)',
  };

  const confidencePct = item.confidence ? Math.round(item.confidence * 100) : null;
  const regionLabel = REGION_NAMES[item.region?.toLowerCase()] || item.region || 'Regional';

  return (
    <Card
      elevation={0}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        borderRadius: '20px',
        bgcolor: '#FFFFFF',
        border: '1px solid rgba(0, 0, 0, 0.07)',
        boxShadow: '0 2px 10px rgba(0, 0, 0, 0.02)',
        overflow: 'hidden',
        transition: 'all 0.28s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:hover': {
          boxShadow: '0 12px 30px rgba(245, 124, 0, 0.14)',
          transform: 'translateY(-3px)',
          borderColor: 'rgba(245, 124, 0, 0.3)',
        },
      }}
    >
      {/* 1. Geometric Image Placeholder Banner */}
      {item.image ? (
        <CardMedia
          component="img"
          height="140"
          image={item.image}
          alt={item.title}
          sx={{ objectFit: 'cover' }}
        />
      ) : (
        <Box
          sx={{
            height: 130,
            background: config.gradient,
            position: 'relative',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden',
          }}
        >
          {/* Subtle background circles for Google Material geometry */}
          <Box
            sx={{
              position: 'absolute',
              width: 140,
              height: 140,
              borderRadius: '50%',
              bgcolor: 'rgba(255, 255, 255, 0.4)',
              top: -30,
              right: -30,
            }}
          />
          <Box
            sx={{
              position: 'absolute',
              width: 90,
              height: 90,
              borderRadius: '50%',
              bgcolor: 'rgba(255, 255, 255, 0.3)',
              bottom: -20,
              left: 10,
            }}
          />

          {/* Centered Domain Placeholder Graphic */}
          <Box
            sx={{
              width: 64,
              height: 64,
              borderRadius: '20px',
              bgcolor: 'rgba(255, 255, 255, 0.85)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.04)',
              backdropFilter: 'blur(4px)',
              zIndex: 1,
            }}
          >
            {config.placeholderIcon}
          </Box>

          {/* Localized Region Badge on Placeholder */}
          <Chip
            label={regionLabel}
            size="small"
            sx={{
              position: 'absolute',
              bottom: 10,
              left: 14,
              fontWeight: 600,
              fontSize: '0.72rem',
              bgcolor: 'rgba(255, 255, 255, 0.92)',
              color: '#374151',
              borderRadius: '12px',
              boxShadow: '0 2px 6px rgba(0, 0, 0, 0.05)',
              zIndex: 1,
            }}
          />

          {/* Match Score Badge on Placeholder */}
          {confidencePct && (
            <Chip
              label={`${confidencePct}% Match`}
              size="small"
              sx={{
                position: 'absolute',
                top: 10,
                right: 14,
                fontWeight: 700,
                fontSize: '0.72rem',
                bgcolor: 'rgba(255, 255, 255, 0.92)',
                color: 'primary.dark',
                borderRadius: '12px',
                boxShadow: '0 2px 6px rgba(0, 0, 0, 0.05)',
                zIndex: 1,
              }}
            />
          )}
        </Box>
      )}

      {/* 2. Card Content */}
      <CardContent sx={{ p: 2.5, flexGrow: 1 }}>
        {/* Category Pill Tag */}
        <Box sx={{ mb: 1.2 }}>
          <Chip
            icon={config.icon}
            label={config.label}
            size="small"
            sx={{
              fontWeight: 600,
              fontSize: '0.75rem',
              color: config.color,
              bgcolor: config.bgColor,
              borderRadius: '16px',
              border: 'none',
              '& .MuiChip-icon': {
                color: config.color,
              },
            }}
          />
        </Box>

        {/* Title */}
        <Typography
          variant="h6"
          component="h3"
          sx={{
            fontWeight: 600,
            fontSize: '1.05rem',
            lineHeight: 1.4,
            color: '#1F2937',
            mb: 0.8,
          }}
        >
          {item.title}
        </Typography>

        {/* Location Pin */}
        {item.location && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1.5, color: '#5F6368' }}>
            <LocationOnOutlinedIcon sx={{ fontSize: 16, color: 'primary.main' }} />
            <Typography variant="caption" sx={{ fontWeight: 500 }}>
              {item.location}
            </Typography>
          </Box>
        )}

        {/* Description */}
        <Typography
          variant="body2"
          sx={{
            color: '#4B5563',
            lineHeight: 1.6,
            mb: 2,
          }}
        >
          {item.description || item.highlights || 'Localized recommendation curated for your regional accent.'}
        </Typography>

        {/* Localized Chips for Tags */}
        {item.tags && item.tags.length > 0 && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.8, mt: 'auto' }}>
            {item.tags.map((tag, idx) => (
              <Chip
                key={idx}
                label={`#${tag}`}
                size="small"
                variant="outlined"
                sx={{
                  borderRadius: '14px',
                  fontSize: '0.72rem',
                  fontWeight: 500,
                  color: '#6B7280',
                  borderColor: 'rgba(0, 0, 0, 0.1)',
                  bgcolor: '#F9FAFB',
                  '&:hover': {
                    bgcolor: 'rgba(245, 124, 0, 0.06)',
                    borderColor: 'primary.light',
                    color: 'primary.dark',
                  },
                }}
              />
            ))}
          </Box>
        )}
      </CardContent>

      {/* 3. Actions / News Article Link Footer */}
      {item.link && (
        <CardActions sx={{ px: 2.5, pb: 2, pt: 0, justifyContent: 'space-between' }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500 }}>
            {item.source || 'Regional RSS'}
          </Typography>
          <Button
            size="small"
            color="primary"
            endIcon={<OpenInNewIcon sx={{ fontSize: 14 }} />}
            href={item.link}
            target="_blank"
            rel="noopener noreferrer"
            sx={{ fontWeight: 600, fontSize: '0.8rem', borderRadius: '16px' }}
          >
            Read Article
          </Button>
        </CardActions>
      )}
    </Card>
  );
}
