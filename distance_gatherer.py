# Author: Connor Anderson
# Last Modified Date: December 4, 2017
from pathlib import Path
from itertools import combinations
import csv
import googlemaps
import genetic_algorithm
import time


class Destination:
    """Model class designed to represent a desired waypoint.
    """
    def __init__(self, country, city):
        self._country = country.title()
        self._city = city.title()

    @property
    def country(self):
        return self._country

    @property
    def city(self):
        return self._city

    def __str__(self):
        return ", ".join((self.city, self.country))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self):
        return hash((self.city, self.country))


class TravelRoute:
    """Model class designed to represent a route between two points.
    """
    def __init__(self, origin, destination, distance, travel_time, units="metric"):
        self._origin = origin
        self._destination = destination
        self._distance = distance
        self._travel_time = travel_time
        if units.lower() == "metric" or units.lower() == "imperial":
            self._units = units

    @property
    def origin(self):
        return self._origin

    @property
    def destination(self):
        return self._destination

    @property
    def distance_m(self):
        if self._units == "imperial":
            return self._distance * 0.3048
        else:
            return self._distance

    @property
    def distance_km(self):
        return self.distance_m / 1000

    @property
    def distance_ft(self):
        if self._units == "metric":
            return self._distance * 3.28084
        else:
            return self._distance

    @property
    def distance_mi(self):
        return self.distance_ft / 5280

    @property
    def travel_time_seconds(self):
        return self._travel_time

    @property
    def travel_time_minutes(self):
        return self._travel_time / 60

    @property
    def travel_time_hours(self):
        return self.travel_time_minutes / 60

    @property
    def travel_time_days(self):
        return self.travel_time_hours / 24

    @property
    def csv_properties(self):
        return [self.origin, self.destination, self.distance_m, self.travel_time_seconds]

    def __str__(self):
        return self.origin + " to " + self._destination + " -> Distance (m): " + str(self.distance_m) + " Time (s): " + str(self.travel_time_seconds)

    def __repr__(self):
        return self.__str__()


class DestinationUtils:
    """Utility class that handles miscellaneous functions (particularly file-based) for the distance_gatherer module.
    """
    @staticmethod
    def parse_destinations_from_csv(filename):
        file = Path(filename)
        destinations = []
        if file.exists():
            with open(filename) as destination_csv:
                destination_reader = csv.reader(destination_csv)
                for destination in destination_reader:
                    destinations.append(Destination(destination[0], destination[1]))
        else:
            raise FileNotFoundError
        return destinations

    @staticmethod
    def write_routes_to_csv(routes, filename="travel_routes.csv"):
        with open(filename, "w") as routes_csv:
            route_writer = csv.writer(routes_csv)
            for route in routes:
                route_writer.writerow(route.csv_properties)

    @staticmethod
    def read_routes_from_csv(filename="travel_routes.csv"):
        destination_route_map = {}
        with open(filename, "r") as routes_csv:
            route_reader = csv.reader(routes_csv)
            for row in route_reader:
                origin_strings = row[0].split(', ')
                destination_strings = row[1].split(', ')
                origin = Destination(city=origin_strings[0], country=origin_strings[1])
                destination = Destination(city=destination_strings[0], country=destination_strings[1])
                route = TravelRoute(origin, destination, int(row[2]), int(row[3]))
                destination_route_map[(origin, destination)] = route
        return destination_route_map

    @staticmethod
    def write_route_to_txt(route, filename="route.txt"):
        with open(filename, "w") as route_txt:
            for destination in route:
                route_txt.write(str(destination) + "\n")

    @staticmethod
    def write_route_maps_url(route):
        maps_url = "https://www.google.com/maps/dir/"
        for destination in route:
            maps_url += str(destination) + "/"
        return maps_url.replace(" ", "+")


class DistanceMatrixApiFacade:
    """Wrapper class that simplifies the usage of the Google Maps Distance Matrix API
    """
    def __init__(self, api_key):
        self._api_key = api_key
        self._gmaps_client = googlemaps.Client(key=api_key)

    def get_driving_routes(self, destinations):
        destination_route_map = {}
        for destination1, destination2 in combinations(destinations, 2):
            try:
                gmaps_response = self._gmaps_client.distance_matrix(origins=str(destination1),
                                                                  destinations=str(destination2),
                                                                  mode="driving",
                                                                  language="English")
                route = TravelRoute(destination1, destination2,
                                    gmaps_response["rows"][0]["elements"][0]["distance"]["value"],
                                    gmaps_response["rows"][0]["elements"][0]["duration"]["value"])
                destination_route_map[(destination1, destination2)] = route
            except Exception as e:
                print("Error retrieving Google Maps info for %s and %s" % str(destination1), str(destination2))
        return destination_route_map


class RouteDistanceFitnessFunction(genetic_algorithm.FitnessFunction):
    """Implementation class that can be used evaluate the fitness of a proposed route on the basis of its distance.
    """
    def __init__(self, destination_route_map):
        self._destination_route_map = destination_route_map
        super(RouteDistanceFitnessFunction, self).__init__()

    def evaluate(self, organism):
        distance_traveled = 0.0
        for locus, destination in enumerate(organism):
            destination1 = organism[locus - 1]
            destination2 = organism[locus]
            if (destination1, destination2) in self._destination_route_map:
                distance_traveled += self._destination_route_map[(destination1, destination2)].distance_m
            elif (destination2, destination1) in self._destination_route_map:
                distance_traveled += self._destination_route_map[(destination2, destination1)].distance_m
            else:
                raise ValueError("Organism contains nonmapped travel route for {0} and {1}".format(str(destination1),
                                                                                                 str(destination2)))
        return distance_traveled


class TravelingSalesmanGeneticAlgorithm(genetic_algorithm.GeneticAlgorithm):
    """Implementation class that implements an approximate solution to the Traveling Salesman Problem using a Genetic Algorithm.
    """
    def create_new_generation(self, current_population):
        selected_organisms = self._selection_function(current_population, self._fitness_function)()
        new_generation = self._crossover_function(selected_organisms,
                                                  len(current_population) - len(selected_organisms))()
        return self._mutation_function(new_generation)()


if __name__ == "__main__":
    """Script designed to use the distance_gatherer and genetic_algorithm modules to demonstrate successful application of
    Genetic Algorithms as they pertain to approximation of the Traveling Salesman Problem.
    """
    generations = 10000
    population_size = 100
    destinations = DestinationUtils.parse_destinations_from_csv('european_capitals.csv')
    # with open("googlemaps_api_key") as api_file:
    #     api_key = api_file.readline()
    # distance_matrix_api = DistanceMatrixApiFacade(api_key)
    # destination_route_map = distance_matrix_api.get_driving_routes(destinations)
    # DestinationUtils.write_routes_to_csv(list(destination_route_map.values()))
    destination_route_map = DestinationUtils.read_routes_from_csv()
    popInitializer = genetic_algorithm.PermutationPopulationInitializer(destinations)
    initial_population = popInitializer.create_population(population_size)

    fitness_function = RouteDistanceFitnessFunction(destination_route_map)
    selection_function = genetic_algorithm.TopTenPercentSelectionFunction
    crossover_function = genetic_algorithm.OrderCrossoverFunction
    mutation_function = genetic_algorithm.ShuffleMutationFunction
    tsp_genetic_algorithm = TravelingSalesmanGeneticAlgorithm(selection_function, fitness_function,
                                                               crossover_function, mutation_function,
                                                               initial_population, generations)
    start_time = time.time()
    optimum_route = tsp_genetic_algorithm.run()
    end_time = time.time()

    filename = "optimum_route_gen{0}_pop{1}.txt".format(str(generations), str(population_size))
    DestinationUtils.write_route_to_txt(optimum_route, filename)
    print("Optimum Route found in {0} seconds, after {1} generations, with population size: {2}.".format(
        str(end_time - start_time), str(generations), str(population_size)))
    print("Route has been written to {0}".format(filename))
