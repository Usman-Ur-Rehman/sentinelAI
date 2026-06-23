'use client'
import { useEffect, useState } from 'react'

export default function Home() {
  const [stats, setStats] = useState({ 
    total_events: 0, 
    total_attacks: 0, 
    alert_rate: 0 
  })

  useEffect(() => {
    const fetchStats = async () => {
      const res = await fetch('http://localhost:8000/stats')
      setStats(await res.json())
    }
    fetchStats()
    const interval = setInterval(fetchStats, 2000)
    return () => clearInterval(interval)
  }, [])

  return (
    <main className='min-h-screen bg-gray-950 text-white p-6'>
      <h1 className='text-2xl font-bold text-indigo-400'>
        AI Threat Detection — Week 1 Scaffold
      </h1>
      <p className='mt-4'>Total Events: {stats.total_events}</p>
      <p>Total Attacks: {stats.total_attacks}</p>
      <p>Alert Rate: {stats.alert_rate}%</p>
    </main>
  )
}