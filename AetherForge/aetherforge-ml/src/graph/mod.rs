//! # Graph Construction and Utilities

use h3o::{CellIndex, LatLng};


pub fn build_h3_graph(
    resolution: u8,
    min_lat: f64,
    min_lon: f64,
    max_lat: f64,
    max_lon: f64,
) -> GraphEdges {
    let mut edges = Vec::new();
    let mut edge_attrs = Vec::new();

    let min_latlng = match h3o::LatLng::new(min_lat, min_lon) {
        Ok(ll) => ll,
        Err(_) => return GraphEdges { edges, edge_attrs },
    };
    let max_latlng = match h3o::LatLng::new(max_lat, max_lon) {
        Ok(ll) => ll,
        Err(_) => return GraphEdges { edges, edge_attrs },
    };

    let h3_res = match h3o::Resolution::try_from(resolution) {
        Ok(r) => r,
        Err(_) => return GraphEdges { edges, edge_attrs },
    };

    let min_cell = min_latlng.to_cell(h3_res);
    let max_cell = max_latlng.to_cell(h3_res);

    let mut cells = Vec::new();
    let mut visited = std::collections::HashSet::new();

    let mut current_opt = Some(min_cell);
    while let Some(current) = current_opt {
        if current <= max_cell && cells.len() < 10000 {
            if visited.insert(u64::from(current)) {
                cells.push(current);
            }
            let neighbors = current.grid_disk::<Vec<CellIndex>>(1);
            current_opt = neighbors
                .iter()
                .find(|&&n| n > min_cell && n <= max_cell && !visited.contains(&u64::from(n)))
                .copied();
        } else {
            current_opt = None;
        }
    }

    for (i, cell) in cells.iter().enumerate() {
        let neighbor_list = cell.grid_disk::<Vec<CellIndex>>(1);
        for n in neighbor_list {
            if let Some(j) = cells.iter().position(|c| u64::from(*c) == u64::from(n)) {
                if i < j {
                    let center1 = LatLng::from(*cell);
                    let center2 = LatLng::from(n);

                    let lat1 = center1.lat();
                    let lon1 = center1.lng();
                    let lat2 = center2.lat();
                    let lon2 = center2.lng();

                    edges.push((i, j));

                    let dist = ((lat1 - lat2).powi(2) as f64 + (lon1 - lon2).powi(2) as f64).sqrt();
                    let diff_lat = (lat1 - lat2) as f32;
                    let diff_lon = (lon1 - lon2) as f32;

                    edge_attrs.push(GraphEdgeAttr {
                        distance: dist as f32,
                        delta_lat: diff_lat,
                        delta_lon: diff_lon,
                    });
                }
            }
        }
    }

    GraphEdges { edges, edge_attrs }
}

#[derive(Debug, Clone)]
pub struct GraphEdges {
    pub edges: Vec<(usize, usize)>,
    pub edge_attrs: Vec<GraphEdgeAttr>,
}

#[derive(Debug, Clone)]
pub struct GraphEdgeAttr {
    pub distance: f32,
    pub delta_lat: f32,
    pub delta_lon: f32,
}

pub fn knn_graph(coords: &[(f64, f64)], k: usize) -> Vec<(usize, usize, f32)> {
    let mut edges = Vec::new();

    for i in 0..coords.len() {
        let mut distances: Vec<(usize, f32)> = coords
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, (lat, lon))| {
                let dist = ((coords[i].0 - lat).powi(2) + (coords[i].1 - lon).powi(2)).sqrt();
                (j, dist as f32)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for &(j, dist) in distances.iter().take(k) {
            edges.push((i, j, dist));
        }
    }

    edges
}

pub fn adjacency_from_edges(num_nodes: usize, edges: &[(usize, usize)]) -> Vec<Vec<f32>> {
    let mut adj = vec![vec![0.0; num_nodes]; num_nodes];

    for &(i, j) in edges {
        if i < num_nodes && j < num_nodes {
            adj[i][j] = 1.0;
            adj[j][i] = 1.0;
        }
    }

    adj
}

pub fn normalized_laplacian(adj: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = adj.len();
    let mut lap = vec![vec![0.0; n]; n];

    let degrees: Vec<f32> = adj.iter().map(|row| row.iter().sum()).collect();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                lap[i][j] = 1.0;
            } else if adj[i][j] > 0.0 && degrees[i] > 0.0 && degrees[j] > 0.0 {
                lap[i][j] = -adj[i][j] / (degrees[i] * degrees[j]).sqrt();
            }
        }
    }

    lap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_graph() {
        let coords = vec![(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)];

        let edges = knn_graph(&coords, 2);
        assert!(edges.len() >= 4);
    }

    #[test]
    fn test_adjacency() {
        let edges = vec![(0, 1), (1, 2)];
        let adj = adjacency_from_edges(3, &edges);
        assert_eq!(adj[0][1], 1.0);
        assert_eq!(adj[1][2], 1.0);
    }
}
