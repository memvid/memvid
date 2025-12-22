#!/usr/bin/env python3
"""
Wikipedia Demo for Memvid

This script demonstrates how to use Memvid with Wikipedia articles as a simple test dataset.
It downloads several Wikipedia articles, creates a video memory, and allows you to chat with the knowledge.

Features:
- Downloads Wikipedia articles from predefined topic lists (100+ articles each)
- Creates optimized QR-encoded video memories with improved density
- Supports semantic search and conversational chat
- Configurable article count, topics, and LLM providers

Usage:
    python examples/wikipedia_demo.py [--num-articles NUM] [--provider PROVIDER] [--topic TOPIC]

Examples:
    # Default: Download 100 AI/tech articles and use Google Gemini
    python examples/wikipedia_demo.py

    # Download 50 articles about science
    python examples/wikipedia_demo.py --num-articles 50 --topic science

    # Use OpenAI instead of Google
    python examples/wikipedia_demo.py --provider openai

    # Create memory only (no chat)
    python examples/wikipedia_demo.py --no-chat

    # List available topics
    python examples/wikipedia_demo.py --list-topics
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for memvid imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  Warning: requests not installed. Run: pip install requests")

from memvid import MemvidEncoder, MemvidChat

# Predefined article topics for testing - expanded for 100+ articles each
TOPIC_ARTICLES = {
    'ai': [
        'Artificial_intelligence',
        'Machine_learning', 
        'Deep_learning',
        'Neural_network',
        'Natural_language_processing',
        'Computer_vision',
        'Large_language_model',
        'Transformer_(machine_learning_model)',
        'GPT-3',
        'Reinforcement_learning',
        'Expert_system',
        'Fuzzy_logic',
        'Genetic_algorithm',
        'Swarm_intelligence',
        'Robotics',
        'Computer_chess',
        'Pattern_recognition',
        'Speech_recognition',
        'Image_recognition',
        'Optical_character_recognition',
        'Automated_reasoning',
        'Knowledge_representation',
        'Semantic_web',
        'Ontology_(information_science)',
        'Bayesian_network',
        'Decision_tree',
        'Support_vector_machine',
        'Random_forest',
        'K-means_clustering',
        'Principal_component_analysis',
        'Artificial_neural_network',
        'Perceptron',
        'Multilayer_perceptron',
        'Convolutional_neural_network',
        'Recurrent_neural_network',
        'Long_short-term_memory',
        'Generative_adversarial_network',
        'Autoencoder',
        'Backpropagation',
        'Gradient_descent',
        'Stochastic_gradient_descent',
        'Adam_(optimization_algorithm)',
        'Overfitting',
        'Cross-validation_(statistics)',
        'Feature_selection',
        'Dimensionality_reduction',
        'Ensemble_learning',
        'Boosting_(machine_learning)',
        'Bagging',
        'AI_alignment',
        'AI_safety',
        'Explainable_artificial_intelligence',
        'Artificial_general_intelligence',
        'Computer_Go',
        'AlphaGo',
        'Watson_(computer)',
        'IBM_Deep_Blue',
        'Turing_test',
        'Chinese_room',
        'Frame_problem',
        'Symbol_grounding_problem',
        'GOFAI',
        'Connectionism',
        'Cybernetics',
        'Computational_intelligence',
        'Soft_computing',
        'Evolutionary_computation',
        'Particle_swarm_optimization',
        'Ant_colony_optimization',
        'Simulated_annealing',
        'Tabu_search',
        'Intelligent_agent',
        'Multi-agent_system',
        'Distributed_artificial_intelligence',
        'Game_theory',
        'Mechanism_design',
        'Auction_theory',
        'Information_theory',
        'Computational_complexity_theory',
        'Algorithm',
        'Data_structure',
        'Graph_theory',
        'Tree_(data_structure)',
        'Hash_table',
        'Dynamic_programming',
        'Greedy_algorithm',
        'Divide_and_conquer',
        'Artificial_life',
        'Cellular_automaton',
        'Complex_system',
        'Emergence',
        'Self-organization',
        'Adaptive_system',
        'Control_theory',
        'Feedback',
        'Homeostasis',
        'Autopoiesis',
        'Systems_theory',
        'Information_system',
        'Database',
        'Data_mining',
        'Big_data',
        'Analytics',
        'Business_intelligence',
        'Predictive_analytics',
        'Recommender_system',
        'Collaborative_filtering',
        'Content-based_filtering'
    ],
    'science': [
        'Physics',
        'Chemistry', 
        'Biology',
        'Quantum_mechanics',
        'Relativity',
        'DNA',
        'Evolution',
        'Periodic_table',
        'Photosynthesis',
        'Climate_change',
        'Thermodynamics',
        'Electromagnetism',
        'Atomic_theory',
        'Molecular_biology',
        'Genetics',
        'Ecology',
        'Biochemistry',
        'Organic_chemistry',
        'Inorganic_chemistry',
        'Physical_chemistry',
        'Analytical_chemistry',
        'Pharmacology',
        'Toxicology',
        'Immunology',
        'Microbiology',
        'Virology',
        'Bacteriology',
        'Mycology',
        'Parasitology',
        'Epidemiology',
        'Pathology',
        'Anatomy',
        'Physiology',
        'Neuroscience',
        'Psychology',
        'Cognitive_science',
        'Behavioral_science',
        'Sociology',
        'Anthropology',
        'Archaeology',
        'Paleontology',
        'Geology',
        'Geophysics',
        'Geochemistry',
        'Meteorology',
        'Oceanography',
        'Hydrology',
        'Soil_science',
        'Environmental_science',
        'Conservation_biology',
        'Zoology',
        'Botany',
        'Entomology',
        'Ornithology',
        'Mammalogy',
        'Herpetology',
        'Ichthyology',
        'Marine_biology',
        'Astrobiology',
        'Astronomy',
        'Astrophysics',
        'Cosmology',
        'Planetary_science',
        'Solar_System',
        'Galaxy',
        'Star',
        'Planet',
        'Exoplanet',
        'Black_hole',
        'Big_Bang',
        'Dark_matter',
        'Dark_energy',
        'Higgs_boson',
        'Standard_Model',
        'Particle_physics',
        'Nuclear_physics',
        'Atomic_physics',
        'Condensed_matter_physics',
        'Solid-state_physics',
        'Plasma_physics',
        'Optics',
        'Laser',
        'Photon',
        'Electron',
        'Proton',
        'Neutron',
        'Quark',
        'Lepton',
        'Boson',
        'Fermion',
        'Quantum_field_theory',
        'String_theory',
        'Loop_quantum_gravity',
        'Theory_of_everything',
        'Unified_field_theory',
        'Supersymmetry',
        'Quantum_entanglement',
        'Quantum_computing',
        'Nanotechnology',
        'Materials_science',
        'Crystallography',
        'Polymer',
        'Ceramic',
        'Metal',
        'Semiconductor',
        'Superconductor',
        'Magnetism',
        'Ferromagnetism'
    ],
    'history': [
        'World_War_II',
        'Ancient_Egypt',
        'Roman_Empire',
        'Renaissance',
        'Industrial_Revolution',
        'American_Civil_War',
        'French_Revolution',
        'Cold_War',
        'Ancient_Greece',
        'Medieval',
        'World_War_I',
        'Byzantine_Empire',
        'Ottoman_Empire',
        'British_Empire',
        'Spanish_Empire',
        'Mongol_Empire',
        'Persian_Empire',
        'Chinese_dynasties',
        'Tang_dynasty',
        'Song_dynasty',
        'Ming_dynasty',
        'Qing_dynasty',
        'Han_dynasty',
        'Roman_Republic',
        'Ancient_Rome',
        'Sparta',
        'Athens',
        'Macedonia',
        'Alexander_the_Great',
        'Julius_Caesar',
        'Augustus',
        'Constantine_I',
        'Charlemagne',
        'William_the_Conqueror',
        'Crusades',
        'Black_Death',
        'Age_of_Exploration',
        'Christopher_Columbus',
        'Vasco_da_Gama',
        'Ferdinand_Magellan',
        'Age_of_Enlightenment',
        'Scientific_Revolution',
        'Protestant_Reformation',
        'Martin_Luther',
        'Thirty_Years_War',
        'English_Civil_War',
        'Glorious_Revolution',
        'American_Revolution',
        'Declaration_of_Independence',
        'George_Washington',
        'Thomas_Jefferson',
        'Abraham_Lincoln',
        'Napoleon',
        'Napoleonic_Wars',
        'Congress_of_Vienna',
        'Revolutions_of_1848',
        'German_unification',
        'Italian_unification',
        'Crimean_War',
        'Franco-Prussian_War',
        'Russian_Revolution',
        'Vladimir_Lenin',
        'Joseph_Stalin',
        'Adolf_Hitler',
        'Winston_Churchill',
        'Franklin_D._Roosevelt',
        'Harry_S._Truman',
        'Dwight_D._Eisenhower',
        'John_F._Kennedy',
        'Martin_Luther_King_Jr.',
        'Nelson_Mandela',
        'Mahatma_Gandhi',
        'Mao_Zedong',
        'Chinese_Civil_War',
        'Korean_War',
        'Vietnam_War',
        'Space_Race',
        'Moon_landing',
        'Berlin_Wall',
        'Cuban_Missile_Crisis',
        'Watergate_scandal',
        'Fall_of_the_Berlin_Wall',
        'Dissolution_of_the_Soviet_Union',
        'September_11_attacks',
        'Iraq_War',
        'Afghanistan_War',
        'Arab_Spring',
        'Ancient_Mesopotamia',
        'Sumerian_civilization',
        'Babylonian_Empire',
        'Assyrian_Empire',
        'Persian_Wars',
        'Peloponnesian_War',
        'Punic_Wars',
        'Germanic_tribes',
        'Viking_Age',
        'Feudalism',
        'Holy_Roman_Empire',
        'Hundred_Years_War',
        'War_of_the_Roses',
        'Spanish_Inquisition',
        'Age_of_Discovery'
    ],
    'space': [
        'Solar_System',
        'Big_Bang',
        'Black_hole',
        'International_Space_Station',
        'Mars',
        'Moon',
        'Milky_Way',
        'Exoplanet',
        'Hubble_Space_Telescope',
        'SpaceX',
        'NASA',
        'Apollo_program',
        'Space_Shuttle',
        'Mercury',
        'Venus',
        'Earth',
        'Jupiter',
        'Saturn',
        'Uranus',
        'Neptune',
        'Pluto',
        'Asteroid',
        'Comet',
        'Meteoroid',
        'Sun',
        'Solar_wind',
        'Solar_flare',
        'Sunspot',
        'Corona',
        'Photosphere',
        'Chromosphere',
        'Nuclear_fusion',
        'Stellar_evolution',
        'Main_sequence',
        'Red_giant',
        'White_dwarf',
        'Neutron_star',
        'Pulsar',
        'Supernova',
        'Planetary_nebula',
        'Nebula',
        'Galaxy_cluster',
        'Dark_matter',
        'Dark_energy',
        'Cosmic_microwave_background',
        'Redshift',
        'Doppler_effect',
        'Parallax',
        'Light-year',
        'Parsec',
        'Astronomical_unit',
        'Kepler_space_telescope',
        'James_Webb_Space_Telescope',
        'Voyager_program',
        'Pioneer_program',
        'Cassini-Huygens',
        'New_Horizons',
        'Mars_rover',
        'Viking_program',
        'Mariner_program',
        'Galileo_(spacecraft)',
        'Juno_(spacecraft)',
        'Parker_Solar_Probe',
        'European_Space_Agency',
        'Roscosmos',
        'JAXA',
        'ISRO',
        'SpaceX_Dragon',
        'Falcon_9',
        'Falcon_Heavy',
        'Starship',
        'Blue_Origin',
        'Virgin_Galactic',
        'Commercial_Crew_Program',
        'Artemis_program',
        'Mars_exploration',
        'Europa_(moon)',
        'Enceladus',
        'Titan_(moon)',
        'Io_(moon)',
        'Ganymede_(moon)',
        'Callisto_(moon)',
        'Phobos_(moon)',
        'Deimos_(moon)',
        'Kuiper_belt',
        'Oort_cloud',
        'Alpha_Centauri',
        'Proxima_Centauri',
        'Betelgeuse',
        'Sirius',
        'Vega',
        'Polaris',
        'Andromeda_Galaxy',
        'Large_Magellanic_Cloud',
        'Small_Magellanic_Cloud',
        'Sagittarius_A*',
        'Event_horizon',
        'Hawking_radiation',
        'Wormhole',
        'General_relativity',
        'Special_relativity'
    ]
}

def download_wikipedia_article(title, max_retries=3):
    """Download a Wikipedia article's content"""
    if not REQUESTS_AVAILABLE:
        return None, "requests library not available"
    
    # Wikipedia API endpoint
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Get the full text from the extract
                content = data.get('extract', '')
                if content:
                    return content, None
                else:
                    return None, "No content in article"
            elif response.status_code == 404:
                return None, f"Article '{title}' not found"
            else:
                return None, f"HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return None, f"Network error: {str(e)}"
            time.sleep(1)  # Wait before retry
    
    return None, "Max retries exceeded"

def download_full_wikipedia_article(title, max_retries=3):
    """Download full Wikipedia article content using the parse API"""
    if not REQUESTS_AVAILABLE:
        return None, "requests library not available"
    
    # Use Wikipedia parse API for full content
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'parse',
        'page': title,
        'format': 'json',
        'prop': 'text',
        'section': 0  # Get all sections
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if 'parse' in data and 'text' in data['parse']:
                    # Extract text from HTML
                    html_content = data['parse']['text']['*']
                    
                    # Simple HTML to text conversion
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Remove unwanted elements
                        for element in soup(['script', 'style', 'table', 'div.navbox', 'div.infobox']):
                            element.decompose()
                        
                        # Get text and clean it up
                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        clean_text = ' '.join(chunk for chunk in chunks if chunk)
                        
                        return clean_text, None
                    except ImportError:
                        # Fallback: use summary API if BeautifulSoup not available
                        return download_wikipedia_article(title)
                else:
                    return None, "No content in API response"
            else:
                return None, f"HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return None, f"Network error: {str(e)}"
            time.sleep(1)
    
    return None, "Max retries exceeded"

def setup_output_dir():
    """Create output directory for demo files"""
    output_dir = Path("output/wikipedia_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def create_wikipedia_memory(articles, topic="ai", use_full_content=True):
    """Create a memory video from Wikipedia articles"""
    print(f"🌍 Wikipedia Demo: Creating memory from {len(articles)} articles")
    print(f"📖 Topic: {topic}")
    print("=" * 60)
    
    output_dir = setup_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memory_name = f"wikipedia_{topic}_{timestamp}"
    
    # Initialize encoder
    encoder = MemvidEncoder()
    
    successful_downloads = 0
    total_content_length = 0
    
    print("\nDownloading articles...")
    
    for i, article_title in enumerate(articles, 1):
        print(f"[{i}/{len(articles)}] {article_title.replace('_', ' ')}...", end=' ')
        
        # Download article content
        if use_full_content:
            content, error = download_full_wikipedia_article(article_title)
        else:
            content, error = download_wikipedia_article(article_title)
        
        if content:
            # Add to encoder (metadata could be added later if supported)
            encoder.add_text(
                content, 
                chunk_size=512,  # Good size for semantic search
                overlap=50
            )
            
            successful_downloads += 1
            total_content_length += len(content)
            print(f"✅ ({len(content):,} chars)")
        else:
            print(f"❌ {error}")
    
    if successful_downloads == 0:
        print("\n❌ No articles downloaded successfully!")
        return None, None, None
    
    print(f"\n📊 Download Summary:")
    print(f"  ✅ Success: {successful_downloads}/{len(articles)} articles")
    print(f"  📝 Total content: {total_content_length:,} characters")
    
    # Get encoder stats
    stats = encoder.get_stats()
    print(f"\n🔤 Processing Stats:")
    print(f"  📦 Total chunks: {stats['total_chunks']}")
    print(f"  📏 Avg chunk size: {stats['avg_chunk_size']:.1f} chars")
    
    # Build video and index
    video_path = output_dir / f"{memory_name}.mp4"
    index_path = output_dir / f"{memory_name}_index.json"
    
    print(f"\n🎬 Building video memory...")
    print(f"  Video: {video_path.name}")
    print(f"  Index: {index_path.name}")
    
    start_time = time.time()
    
    try:
        build_stats = encoder.build_video(str(video_path), str(index_path), show_progress=True)
        build_time = time.time() - start_time
        
        print(f"\n✅ Memory created successfully in {build_time:.1f}s")
        print(f"🎥 Video: {build_stats['video_size_mb']:.1f} MB, {build_stats['duration_seconds']:.1f}s")
        print(f"📇 Index: {build_stats['index_stats']['total_chunks']} chunks")
        
        # Save metadata about this demo
        demo_metadata = {
            "created": datetime.now().isoformat(),
            "topic": topic,
            "articles_requested": len(articles),
            "articles_successful": successful_downloads,
            "total_content_chars": total_content_length,
            "build_stats": build_stats,
            "article_titles": [a.replace('_', ' ') for a in articles[:successful_downloads]]
        }
        
        metadata_path = output_dir / f"{memory_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(demo_metadata, f, indent=2)
        
        return str(video_path), str(index_path), demo_metadata
        
    except Exception as e:
        print(f"\n❌ Failed to create memory: {e}")
        return None, None, None

def start_chat_session(video_path, index_path, provider='google'):
    """Start an interactive chat session with the Wikipedia memory"""
    print(f"\n💬 Starting chat session with Wikipedia memory")
    print(f"🤖 LLM Provider: {provider}")
    print("=" * 60)
    
    try:
        chat = MemvidChat(video_path, index_path, provider=provider)
        
        print(f"\n🎯 Try asking questions like:")
        print(f"  • 'Tell me about artificial intelligence'")
        print(f"  • 'What are neural networks?'")
        print(f"  • 'Explain machine learning'")
        print(f"  • 'What is the history of computing?'")
        print(f"\n💡 Type 'quit' or 'exit' to end the session")
        print("-" * 60)
        
        while True:
            query = input("\n🤔 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print("🔍 Searching...", end='', flush=True)
            start_time = time.time()
            
            try:
                response = chat.chat(query)
                response_time = time.time() - start_time
                
                print(f"\r🤖 Response ({response_time:.1f}s):")
                print("-" * 40)
                print(response)
                print("-" * 40)
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("💡 Make sure you have set up your API key for the chosen provider")
    
    except Exception as e:
        print(f"❌ Failed to start chat: {e}")
        return False
    
    print("\n👋 Chat session ended. Thanks for trying Memvid!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Wikipedia Demo for Memvid")
    parser.add_argument('--num-articles', type=int, default=100, 
                       help='Number of articles to download (default: 100)')
    parser.add_argument('--topic', choices=list(TOPIC_ARTICLES.keys()), default='ai',
                       help='Topic area for articles (default: ai)')
    parser.add_argument('--provider', choices=['openai', 'google', 'anthropic'], default='google',
                       help='LLM provider for chat (default: google)')
    parser.add_argument('--no-chat', action='store_true',
                       help='Only create memory, skip chat session')
    parser.add_argument('--full-content', action='store_true', default=True,
                       help='Download full article content (default: True)')
    parser.add_argument('--list-topics', action='store_true',
                       help='List available topics and exit')
    
    args = parser.parse_args()
    
    if args.list_topics:
        print("Available topics:")
        for topic, articles in TOPIC_ARTICLES.items():
            print(f"  {topic}: {len(articles)} articles")
            for article in articles[:3]:
                print(f"    - {article.replace('_', ' ')}")
            if len(articles) > 3:
                print(f"    ... and {len(articles) - 3} more")
        return
    
    if not REQUESTS_AVAILABLE:
        print("❌ Error: This demo requires the 'requests' library")
        print("💡 Install it with: pip install requests")
        print("💡 For HTML parsing (recommended): pip install requests beautifulsoup4")
        return 1
    
    # Get articles for the chosen topic
    available_articles = TOPIC_ARTICLES[args.topic]
    articles_to_download = available_articles[:args.num_articles]
    
    print("🌟 Memvid Wikipedia Demo")
    print("=" * 60)
    print(f"📖 Topic: {args.topic}")
    print(f"📚 Articles: {len(articles_to_download)}")
    print(f"🤖 LLM: {args.provider}")
    
    # Create memory from Wikipedia articles  
    video_path, index_path, metadata = create_wikipedia_memory(
        articles_to_download, 
        args.topic,
        args.full_content
    )
    
    if not video_path:
        print("❌ Failed to create memory. Exiting.")
        return 1
    
    # Show some interesting stats
    if metadata:
        print(f"\n📈 Final Stats:")
        print(f"  🎬 Video file: {Path(video_path).stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  📇 Index file: {Path(index_path).stat().st_size / 1024:.1f} KB")
        print(f"  🧠 Knowledge: {metadata['articles_successful']} Wikipedia articles")
    
    # Start chat session unless disabled
    if not args.no_chat:
        success = start_chat_session(video_path, index_path, args.provider)
        return 0 if success else 1
    else:
        print(f"\n✅ Memory created successfully!")
        print(f"💡 To chat with it later, run:")
        print(f"   python -c \"from memvid import chat_with_memory; chat_with_memory('{video_path}', '{index_path}')\"")
    
    return 0

if __name__ == "__main__":
    exit(main()) 